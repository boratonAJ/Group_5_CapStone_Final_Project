"""
models.py — Model training, serialization, and evaluation helpers.
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, average_precision_score,
    confusion_matrix, precision_score, recall_score, f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create an OneHotEncoder that works across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical columns from a feature frame."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(
    X: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
):
    """Build a preprocessing transformer for mixed-type tabular data."""
    if numeric_features is None or categorical_features is None:
        numeric_features, categorical_features = infer_feature_types(X)

    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ])
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        return "passthrough"

    return ColumnTransformer(transformers)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    C: float = 0.1,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """
    Train a logistic regression pipeline with mixed-type preprocessing.
    Interpretable baseline model.
    """
    preprocessor = build_preprocessor(X_train)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("lr", LogisticRegression(
            C=C, max_iter=max_iter, random_state=random_state,
            class_weight="balanced", solver="lbfgs",
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_gbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict = None,
    random_state: int = 42,
):
    """
    Train an XGBoost gradient-boosted model with shared preprocessing.
    Candidate deployment model.
    """
    import xgboost as xgb

    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 50,
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "random_state": random_state,
            "eval_metric": "auc",
            "n_jobs": -1,
            "verbosity": 0,
        }

    preprocessor = build_preprocessor(X_train)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("gbm", xgb.XGBClassifier(**params)),
    ])
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        print(f"[GBM] Validation AUC: {val_auc:.4f}")

    return model


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
    split_name: str = "Test",
) -> dict:
    """
    Compute a comprehensive set of evaluation metrics.

    Returns
    -------
    dict: AUC, PR-AUC, Brier, KS, accuracy, precision, recall, F1, confusion matrix
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    brier = brier_score_loss(y, y_prob)

    # KS statistic
    from scipy.stats import ks_2samp
    pos_probs = y_prob[y == 1]
    neg_probs = y_prob[y == 0]
    ks_stat, _ = ks_2samp(pos_probs, neg_probs)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": model_name,
        "split": split_name,
        "threshold": threshold,
        "n": len(y),
        "auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier": round(brier, 4),
        "ks": round(ks_stat, 4),
        "accuracy": round((tp + tn) / len(y), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "positive_rate_predicted": round(y_pred.mean(), 4),
        "positive_rate_actual": round(y.mean(), 4),
    }

    print(f"\n{'='*55}")
    print(f"  {model_name} — {split_name} Set Metrics")
    print(f"{'='*55}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"  KS:        {metrics['ks']:.4f}")
    print(f"  Brier:     {metrics['brier']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}  (threshold={threshold})")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  CM: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"{'='*55}")
    return metrics


def save_model(model, model_name: str, models_dir: str = "models") -> str:
    """Save model artifact with versioned filename and metadata JSON."""
    os.makedirs(models_dir, exist_ok=True)
    version = datetime.now().strftime("%Y%m%d")
    model_path = os.path.join(models_dir, f"{model_name}_v{version}.pkl")
    meta_path = os.path.join(models_dir, f"{model_name}_v{version}_meta.json")

    joblib.dump(model, model_path)
    meta = {
        "model_name": model_name,
        "version": version,
        "model_class": type(model).__name__,
        "created_at": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[save] Model saved: {model_path}")
    print(f"[save] Meta saved:  {meta_path}")
    return model_path


def select_threshold(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    strategy: str = "f1",
) -> float:
    """
    Select operating threshold using one of three strategies:
        'f1'       : maximize F1 on validation set
        'precision_recall_balance' : where precision ≈ recall
        'approval_rate' : match observed approval rate in validation set

    Returns the selected threshold (float).
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_t = 0.5
    best_score = -1

    if strategy == "f1":
        for t in thresholds:
            y_pred = (y_prob_val >= t).astype(int)
            score = f1_score(y_val, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_t = t

    elif strategy == "approval_rate":
        target_rate = y_val.mean()
        best_t = np.percentile(y_prob_val, (1 - target_rate) * 100)

    return round(float(best_t), 4)
