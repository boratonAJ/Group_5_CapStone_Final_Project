from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.engineering import build_preprocessor
from src.models.evaluate import classification_metrics


@dataclass
class SplitData:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def make_splits(x: pd.DataFrame, y: pd.Series, random_state: int = 42) -> SplitData:
    """Create stratified train/validation/test splits."""

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.4,
        random_state=random_state,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )
    return SplitData(x_train, x_val, x_test, y_train, y_val, y_test)


def _optimal_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    best_t = 0.5
    best_j = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


def build_models(random_state: int = 42) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            random_state=random_state,
            solver="saga",
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=None,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def train_model(
    model_name: str,
    estimator,
    split: SplitData,
) -> dict:
    """Train one model and return trained pipeline and metrics."""

    preprocessor = build_preprocessor(split.x_train)
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

    pipe.fit(split.x_train, split.y_train)

    p_train = pipe.predict_proba(split.x_train)[:, 1]
    p_val = pipe.predict_proba(split.x_val)[:, 1]
    p_test = pipe.predict_proba(split.x_test)[:, 1]

    threshold = _optimal_threshold(split.y_val, p_val)

    yhat_train = (p_train >= threshold).astype(int)
    yhat_val = (p_val >= threshold).astype(int)
    yhat_test = (p_test >= threshold).astype(int)

    return {
        "model_name": model_name,
        "pipeline": pipe,
        "threshold": threshold,
        "pred": {
            "train": {"y_prob": p_train, "y_pred": yhat_train},
            "val": {"y_prob": p_val, "y_pred": yhat_val},
            "test": {"y_prob": p_test, "y_pred": yhat_test},
        },
        "metrics": {
            "train": classification_metrics(split.y_train, yhat_train, p_train),
            "val": classification_metrics(split.y_val, yhat_val, p_val),
            "test": classification_metrics(split.y_test, yhat_test, p_test),
        },
    }
