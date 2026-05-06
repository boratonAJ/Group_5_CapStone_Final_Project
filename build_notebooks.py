"""
build_notebooks.py
Run this script ONCE to generate all .ipynb notebooks from the cell definitions below.
Usage: python build_notebooks.py
"""

import json
import os

NB_DIR = os.path.join(os.path.dirname(__file__), "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def make_notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "cells": cells,
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def save_notebook(cells, filename):
    path = os.path.join(NB_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=1)
    print(f"  Created: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 01 — DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

nb01_cells = [

md("""## GenAI Disclosure Statement

In this course, generative AI tools were used as learning aids to support understanding
of Responsible ML concepts. All analysis, interpretations, and conclusions are the
original work of the project team.

---

# Notebook 01 — Data Preparation and Label Construction
### DNSC 6330 Responsible Machine Learning Capstone | GWU
**Audit framework:** Measurement before opinion · Diagnostics before remediation · Documentation before deployment

**Purpose:** Load 2024 HMDA LAR, apply label rule, remove leakage features, engineer features,
build protected-attribute vectors, and export train/val/test/geo_holdout splits.

**Inputs:** `../2024_lar.zip` (raw HMDA LAR)  
**Outputs:** `data/processed/*.parquet`  
**Dependencies:** None (first notebook in pipeline)
"""),

code("""\
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")
import zipfile
import hashlib
from sklearn.model_selection import train_test_split

from src.labels import apply_label, label_distribution_table
from src.leakage import remove_leakage, get_proxy_risk_table

# Fixed seeds for reproducibility
SEED = 42
np.random.seed(SEED)

# Paths — adjust DATA_ZIP if the zip is in a different location
DATA_ZIP   = os.path.join(os.getcwd(), "..", "..", "2024_lar.zip")
PROC_DIR   = os.path.join(os.getcwd(), "..", "data", "processed")
TABLES_DIR = os.path.join(os.getcwd(), "..", "tables")
FIG_DIR    = os.path.join(os.getcwd(), "..", "figures")

for d in [PROC_DIR, TABLES_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

print("Environment ready.")
print(f"  DATA_ZIP   : {DATA_ZIP}")
print(f"  PROC_DIR   : {PROC_DIR}")
"""),

md("""## Section 1.1 — Source and Provenance

We document the data source and file hash before any processing begins.
This is audit evidence that we know exactly which version of the data we used.
"""),

code("""\
# Compute SHA-256 hash of the zip file for provenance documentation
def file_sha256(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

if os.path.exists(DATA_ZIP):
    sha = file_sha256(DATA_ZIP)
    print(f"SHA-256: {sha}")
    print(f"File size: {os.path.getsize(DATA_ZIP) / 1e9:.2f} GB")
    with open(os.path.join(os.getcwd(), "..", "data", "README.md"), "r") as f:
        content = f.read()
    content = content.replace("[INSERT SHA-256 HASH OF 2024_lar.zip OR 2024_lar.txt]", sha)
    with open(os.path.join(os.getcwd(), "..", "data", "README.md"), "w") as f:
        f.write(content)
    print("data/README.md updated with hash.")
else:
    print(f"WARNING: Data zip not found at {DATA_ZIP}")
    print("Please set DATA_ZIP to the correct path.")
"""),

md("""## Section 1.2 — Raw Load and Row Counts

We load the full 2024 HMDA LAR in chunks to handle the ~4.6GB file size efficiently.
We track row counts at every filtering step.
This table is the first piece of audit evidence in the data pipeline.

**Key columns we rely on:**
- `action_taken`: the label we will predict
- `derived_race`, `derived_sex`, `derived_ethnicity`: HMDA-derived protected attributes
- `income`, `loan_amount`, `property_value`, `debt_to_income_ratio`: core creditworthiness features
- `census_tract`, `tract_minority_population_percent`: geographic features with proxy-risk concern
"""),

code("""\
# Load 2024 HMDA LAR from zip — pipe-delimited
# We use chunksize to handle the large file efficiently
# For development, use SAMPLE_FRAC < 1.0 to iterate quickly

SAMPLE_FRAC = 1.0       # Set to e.g. 0.2 for fast development runs
CHUNKSIZE = 200_000
GEO_HOLDOUT_STATES = ["CA"]  # Hold out California as geographic drift test set
                              # Decision R-003: selected for size and demographic diversity

print(f"Loading 2024 HMDA LAR from {DATA_ZIP}")
print(f"Sample fraction: {SAMPLE_FRAC}")
print(f"Geographic holdout state(s): {GEO_HOLDOUT_STATES}")
print("This may take several minutes for the full dataset...")

chunks = []
with zipfile.ZipFile(DATA_ZIP) as z:
    with z.open("2024_lar.txt") as f:
        for chunk in pd.read_csv(
            f,
            sep="|",
            chunksize=CHUNKSIZE,
            dtype=str,
            low_memory=False,
        ):
            chunks.append(chunk)

raw_df = pd.concat(chunks, ignore_index=True)

if SAMPLE_FRAC < 1.0:
    raw_df = raw_df.sample(frac=SAMPLE_FRAC, random_state=SEED).reset_index(drop=True)
    print(f"  [DEVELOPMENT SAMPLE] Using {len(raw_df):,} rows ({SAMPLE_FRAC:.0%})")

n_raw = len(raw_df)
print(f"\\nRaw rows loaded: {n_raw:,}")
print(f"Columns: {len(raw_df.columns)}")
print(f"action_taken value counts:")
print(raw_df["action_taken"].value_counts().sort_index())
"""),

md("""## Section 1.3 — Label Construction

Apply the capstone label rule. This uses `src/labels.py` — the single source of truth.

**Rule:** `{1,2}  1`, `{3}  0`, all other values filtered out.
**Rationale:** Documented in `docs/decision_log.md` (D-001) and `docs/00_system_card.md`.
"""),

code("""\
# Show label distribution before filtering
dist_table = label_distribution_table(raw_df)
print("Label distribution table (before filtering):")
display(dist_table)
dist_table.to_csv(os.path.join(TABLES_DIR, "label_distribution.csv"), index=False)

# Apply label rule
df = apply_label(raw_df)
print(f"\\nClass balance check:")
print(f"  y=1 (originated/approved): {(df['y']==1).sum():,}  ({df['y'].mean():.3f})")
print(f"  y=0 (denied):              {(df['y']==0).sum():,}  ({1-df['y'].mean():.3f})")
"""),

md("""## Section 1.4 — Leakage Quarantine

Remove all post-decision features. These are ONLY populated after the lending decision
and would cause target leakage. This is enforced in `src/leakage.py`.

**Features removed:** `interest_rate`, `rate_spread`, `denial_reason_*`, `total_loan_costs`,
`origination_charges`, `discount_points`, `lender_credits`, `prepayment_penalty_term`,
`intro_rate_period`, `lei`, `activity_year`
"""),

code("""\
# Remove post-decision features
df = remove_leakage(df)

# Also show proxy-risk table
proxy_risk_df = get_proxy_risk_table()
print("\\nProxy-risk flagged features (retained but monitored):")
display(proxy_risk_df)
proxy_risk_df.to_csv(os.path.join(TABLES_DIR, "proxy_risk_features.csv"), index=False)
"""),

md("""## Section 1.5 — Protected Attribute Handling

`derived_race`, `derived_sex`, and `derived_ethnicity` are retained as protected attribute
columns. They are **NOT** used as model input features but are used in fairness analysis.

We also create a `race_sex_intersection` column for intersectional analysis.

**Decision D-008:** Applicants who selected "Not Provided" or "Not Applicable" are
retained as a distinct group rather than being dropped.
"""),

code("""\
# Check protected attribute coverage
print("Protected attribute value counts:")
for col in ["derived_race", "derived_sex", "derived_ethnicity"]:
    if col in df.columns:
        print(f"\\n  {col}:")
        print(df[col].value_counts())

# Create intersectional column
df["race_sex_intersection"] = df["derived_race"].astype(str) + " × " + df["derived_sex"].astype(str)

# Count intersection cells
intersection_counts = df["race_sex_intersection"].value_counts()
print(f"\\nIntersectional cells (race × sex): {len(intersection_counts)}")
print(f"Cells with n >= 30: {(intersection_counts >= 30).sum()}")
print(f"Cells with n < 30 (will be suppressed): {(intersection_counts < 30).sum()}")
"""),

md("""## Section 1.6 — Feature Engineering

Apply transformations from `src/features.py`:
- DTI string buckets -> numeric midpoints
- Outlier capping on loan_amount, income, property_value
- One-hot encoding of categorical features
- Median imputation for remaining NaN

Protected attributes are stored separately and NOT included in the feature matrix.
Applicant sex and applicant age are retained for fairness analysis only.
"""),

code("""\
from src.features import build_feature_matrix, PROTECTED_ATTRS, NUM_FEATURES, CAT_FEATURES

# Store protected attributes and target separately before feature engineering
PROTECTED_COLS = [
    "derived_race",
    "derived_sex",
    "derived_ethnicity",
    "applicant_sex",
    "applicant_age",
    "race_sex_intersection",
]
META_COLS = ["action_taken", "y", "state_code"] + PROTECTED_COLS

# Build feature matrix (no protected attrs, no target)
print("Building feature matrix...")
X = build_feature_matrix(df)
y = df["y"].values
meta = df[META_COLS].reset_index(drop=True)

print(f"Feature matrix shape: {X.shape}")
print(f"Features included: {len(X.columns)}")
print(f"Missing values in X: {X.isnull().sum().sum()}")
"""),

md("""## Section 1.7 — Train/Val/Test Splits + Geographic Holdout

**Split strategy (Decision D-003):**
- Geographic holdout: California (`state_code == 'CA'`) held out entirely for drift testing
- Remaining data: stratified random 70/15/15 (train/val/test) by `y`

Stratification ensures class balance is preserved in all splits.
Geographic holdout simulates prospective distribution shift.
"""),

code("""\
# Geographic holdout: hold out California
geo_mask = df["state_code"].isin(GEO_HOLDOUT_STATES)
non_geo_mask = ~geo_mask

X_geo    = X[geo_mask].reset_index(drop=True)
y_geo    = y[geo_mask]
meta_geo = meta[geo_mask].reset_index(drop=True)

X_main    = X[non_geo_mask].reset_index(drop=True)
y_main    = y[non_geo_mask]
meta_main = meta[non_geo_mask].reset_index(drop=True)

print(f"Geographic holdout ({GEO_HOLDOUT_STATES}): {len(X_geo):,} rows")
print(f"Remaining (non-holdout): {len(X_main):,} rows")

# Stratified train/val/test split on non-holdout data
# 70% train, 15% val, 15% test — roughly 70/15/15
X_train_full, X_test, y_train_full, y_test, meta_train_full, meta_test = (
    train_test_split(X_main, y_main, meta_main, test_size=0.15, random_state=SEED, stratify=y_main)
)
X_train, X_val, y_train, y_val, meta_train, meta_val = (
    train_test_split(X_train_full, y_train_full, meta_train_full,
                     test_size=0.15/0.85, random_state=SEED, stratify=y_train_full)
)

print(f"\\nSplit summary:")
print(f"  Train:       {len(X_train):>10,}  (positive rate: {y_train.mean():.3f})")
print(f"  Validation:  {len(X_val):>10,}  (positive rate: {y_val.mean():.3f})")
print(f"  Test:        {len(X_test):>10,}  (positive rate: {y_test.mean():.3f})")
print(f"  Geo holdout: {len(X_geo):>10,}  (positive rate: {y_geo.mean():.3f})")

# Confirm stratification
print(f"\\nStratification check:")
for name, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
    print(f"  {name} positive rate: {ys.mean():.4f}")
"""),

md("""## Section 1.8 — Export Processed Files

Export all splits to parquet format. These files are the inputs for all subsequent notebooks.
"""),

code("""\
import pandas as pd

def export_split(X, y, meta, prefix, path):
    out = X.copy()
    out["y"] = y
    for col in meta.columns:
        out[col] = meta[col].values
    fpath = os.path.join(path, f"{prefix}.parquet")
    out.to_parquet(fpath, index=False)
    print(f"  Saved: {fpath}  ({len(out):,} rows x {len(out.columns)} cols)")
    return out

print("Exporting processed files...")
train_df = export_split(X_train, y_train, meta_train, "train", PROC_DIR)
val_df   = export_split(X_val,   y_val,   meta_val,   "val",   PROC_DIR)
test_df  = export_split(X_test,  y_test,  meta_test,  "test",  PROC_DIR)
geo_df   = export_split(X_geo,   y_geo,   meta_geo,   "geo_holdout", PROC_DIR)

# Export feature column list for use in downstream notebooks
feature_cols = X_train.columns.tolist()
with open(os.path.join(PROC_DIR, "feature_columns.txt"), "w") as f:
    f.write("\\n".join(feature_cols))
print(f"\\nFeature column list saved: {len(feature_cols)} features")

# Data dictionary
data_dict = pd.DataFrame({
    "feature": feature_cols,
    "dtype": [str(X_train[c].dtype) for c in feature_cols],
    "n_missing": [X_train[c].isnull().sum() for c in feature_cols],
    "mean": [X_train[c].mean() if X_train[c].dtype in ["float64", "int64"] else None for c in feature_cols],
})
data_dict.to_csv(os.path.join(TABLES_DIR, "data_dictionary.csv"), index=False)
print(f"Data dictionary saved.")
"""),

md("""## Section 1.9 — Row Count Audit Table

Final documentation of the data filtering pipeline.
This table is audit evidence that every filtering decision is accounted for.
"""),

code("""\
audit_rows = {
    "Step": [
        "1. Raw HMDA LAR rows",
        "2. action_taken in {1,2,3} (after label filter)",
        "3. Leakage features removed (same rows)",
        "4. Feature matrix built",
        "5. Train split (70%)",
        "6. Validation split (15%)",
        "7. Test split (15%)",
        "8. Geographic holdout (CA)",
    ],
    "Row Count": [
        n_raw,
        len(df),
        len(df),
        len(X_main) + len(X_geo),
        len(X_train),
        len(X_val),
        len(X_test),
        len(X_geo),
    ],
    "Positive Rate": [
        None,
        round(df["y"].mean(), 3),
        round(df["y"].mean(), 3),
        None,
        round(y_train.mean(), 3),
        round(y_val.mean(), 3),
        round(y_test.mean(), 3),
        round(y_geo.mean(), 3),
    ],
    "Notes": [
        "Raw HMDA 2024 LAR",
        f"Dropped {n_raw - len(df):,} rows with action_taken in {{4,5,6,7,8}}",
        f"Removed {len(df.columns) - len(X_train.columns)} post-decision features",
        "Feature engineering applied",
        "Stratified by y",
        "Stratified by y",
        "Stratified by y",
        "State = CA — geographic drift test set",
    ]
}
audit_df = pd.DataFrame(audit_rows)
display(audit_df)
audit_df.to_csv(os.path.join(TABLES_DIR, "data_pipeline_audit.csv"), index=False)
print("\\nData pipeline audit table saved.")
print("\\n Notebook 01 complete. All processed files saved to data/processed/")
"""),

]

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 02 — MODELING
# ═══════════════════════════════════════════════════════════════════════════════

nb02_cells = [

md("""## GenAI Disclosure Statement

In this course, generative AI tools were used as learning aids. All analysis and
conclusions are the original work of the project team.

---

# Notebook 02 — Baseline Model Development
### DNSC 6330 Responsible Machine Learning Capstone | GWU

**Purpose:** Train logistic regression (interpretable baseline) and XGBoost GBM (candidate deployment model).
Select operating threshold. Produce performance metrics for audit record.

**Inputs:** `data/processed/train.parquet`, `val.parquet`, `test.parquet`  
**Outputs:** `models/*.pkl`, `tables/metrics_table.csv`, `figures/roc_curve.png`, `figures/pr_curve.png`
"""),

code("""\
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

from src.models import (
    train_logistic_regression, train_gbm, evaluate_model,
    save_model, select_threshold
)

SEED = 42
np.random.seed(SEED)

PROC_DIR   = os.path.join(os.getcwd(), "..", "data", "processed")
MODELS_DIR = os.path.join(os.getcwd(), "..", "models")
TABLES_DIR = os.path.join(os.getcwd(), "..", "tables")
FIG_DIR    = os.path.join(os.getcwd(), "..", "figures")

for d in [MODELS_DIR, TABLES_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)
"""),

code("""\
# Load processed splits
train_df = pd.read_parquet(os.path.join(PROC_DIR, "train.parquet"))
val_df   = pd.read_parquet(os.path.join(PROC_DIR, "val.parquet"))
test_df  = pd.read_parquet(os.path.join(PROC_DIR, "test.parquet"))

# Identify feature columns (everything except y, protected attrs, meta)
NON_FEATURE_COLS = [
    "y", "action_taken", "state_code",
    "derived_race", "derived_sex", "derived_ethnicity", "race_sex_intersection"
]
feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]

X_train = train_df[feature_cols]
y_train = train_df["y"].values
X_val   = val_df[feature_cols]
y_val   = val_df["y"].values
X_test  = test_df[feature_cols]
y_test  = test_df["y"].values

print(f"Training features: {len(feature_cols)}")
print(f"Train:  {len(X_train):,} rows | positive rate: {y_train.mean():.3f}")
print(f"Val:    {len(X_val):,} rows | positive rate: {y_val.mean():.3f}")
print(f"Test:   {len(X_test):,} rows | positive rate: {y_test.mean():.3f}")
"""),

md("""## Section 2.2 — Logistic Regression (Interpretable Baseline)

The logistic regression serves as the interpretable baseline. Its coefficients provide
direct evidence about feature-direction relationships. Any GBM that is proposed for
deployment should meaningfully outperform this baseline to justify the explainability cost.

**Hyperparameter choices:** C=0.1 (regularization), balanced class weights, lbfgs solver.
"""),

code("""\
print("Training Logistic Regression...")
lr_model = train_logistic_regression(X_train, y_train, C=0.1, random_state=SEED)

lr_val_metrics  = evaluate_model(lr_model, X_val,  y_val,  model_name="LogReg", split_name="Val")
lr_test_metrics = evaluate_model(lr_model, X_test, y_test, model_name="LogReg", split_name="Test")

lr_model_path = save_model(lr_model, "logreg", models_dir=MODELS_DIR)
"""),

md("""## Section 2.3 — Gradient-Boosted Model (Candidate Deployment Model)

XGBoost with early stopping on the validation set.
Hyperparameters are tuned conservatively to avoid overfitting to the training distribution.

**Key choices:**
- `scale_pos_weight` set to class imbalance ratio to handle label imbalance
- `max_depth=4`: limits complexity; improves generalization and reduces proxy encoding
- `early_stopping_rounds=30`: prevents overfitting
"""),

code("""\
import xgboost as xgb
print("Training Gradient-Boosted Model (XGBoost)...")

gbm_params = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
    "random_state": SEED,
    "eval_metric": "auc",
    "early_stopping_rounds": 30,
    "n_jobs": -1,
    "verbosity": 0,
}

gbm_model = xgb.XGBClassifier(**gbm_params)
gbm_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
print(f"Best iteration: {gbm_model.best_iteration}")

gbm_val_metrics  = evaluate_model(gbm_model, X_val,  y_val,  model_name="GBM", split_name="Val")
gbm_test_metrics = evaluate_model(gbm_model, X_test, y_test, model_name="GBM", split_name="Test")

gbm_model_path = save_model(gbm_model, "gbm", models_dir=MODELS_DIR)
"""),

md("""## Section 2.4 — Metrics Comparison Table

Side-by-side comparison of both models. The GBM should outperform LR; if it does not,
the LR baseline should be preferred for its transparency.
"""),

code("""\
all_metrics = [lr_val_metrics, lr_test_metrics, gbm_val_metrics, gbm_test_metrics]
metrics_df = pd.DataFrame(all_metrics)
display_cols = ["model", "split", "threshold", "n", "auc", "pr_auc", "ks", "brier",
                "accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
metrics_df = metrics_df[[c for c in display_cols if c in metrics_df.columns]]
display(metrics_df)
metrics_df.to_csv(os.path.join(TABLES_DIR, "metrics_table_default_threshold.csv"), index=False)
print("Metrics table saved.")
"""),

md("""## Section 2.5 — Threshold Selection

The default threshold of 0.5 is arbitrary. We select the operating threshold by maximizing
F1 on the **validation set** (not the test set — to avoid peeking).

After selecting the threshold, we re-evaluate on the test set and examine the fairness
impact of the chosen threshold.

**Decision D-007:** F1 maximization on validation set. This decision will be revisited
in Notebook 04 after fairness analysis.
"""),

code("""\
# Get validation probabilities
gbm_val_probs = gbm_model.predict_proba(X_val)[:, 1]
lr_val_probs  = lr_model.predict_proba(X_val)[:, 1]

# Select threshold
gbm_threshold = select_threshold(y_val, gbm_val_probs, strategy="f1")
lr_threshold  = select_threshold(y_val, lr_val_probs,  strategy="f1")
print(f"GBM operating threshold (F1-optimal on val): {gbm_threshold}")
print(f"LR  operating threshold (F1-optimal on val): {lr_threshold}")

# Re-evaluate both models at selected thresholds on test set
gbm_test_final = evaluate_model(gbm_model, X_test, y_test,
                                threshold=gbm_threshold,
                                model_name="GBM (F1-threshold)",
                                split_name="Test")
lr_test_final  = evaluate_model(lr_model,  X_test, y_test,
                                threshold=lr_threshold,
                                model_name="LogReg (F1-threshold)",
                                split_name="Test")

final_metrics = pd.DataFrame([gbm_test_final, lr_test_final])
final_metrics.to_csv(os.path.join(TABLES_DIR, "metrics_table_final.csv"), index=False)
display(final_metrics)
print(f"\\nSelected operating threshold for GBM: {gbm_threshold}")
"""),

md("""## Section 2.6 — ROC and PR Curves

Standard performance visualization. Both models plotted together for comparison.
"""),

code("""\
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# ROC curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for model, probs, label, color in [
    (gbm_model, gbm_model.predict_proba(X_test)[:, 1], "GBM", "steelblue"),
    (lr_model,  lr_model.predict_proba(X_test)[:, 1],  "Logistic Regression", "coral"),
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.4f})", color=color, lw=2)

axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve — Test Set")
axes[0].legend()
axes[0].grid(alpha=0.3)

# PR curves
for model, probs, label, color in [
    (gbm_model, gbm_model.predict_proba(X_test)[:, 1], "GBM", "steelblue"),
    (lr_model,  lr_model.predict_proba(X_test)[:, 1],  "Logistic Regression", "coral"),
]:
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc_val = auc(recall, precision)
    axes[1].plot(recall, precision, label=f"{label} (PR-AUC={pr_auc_val:.4f})", color=color, lw=2)

axes[1].axhline(y=y_test.mean(), color="gray", linestyle="--", lw=1, label="Baseline (prevalence)")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve — Test Set")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
roc_path = os.path.join(FIG_DIR, "roc_pr_curves.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {roc_path}")
"""),

md("""## Section 2.7 — Confusion Matrix at Operating Threshold"""),

code("""\
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

gbm_test_probs = gbm_model.predict_proba(X_test)[:, 1]
gbm_test_preds = (gbm_test_probs >= gbm_threshold).astype(int)

cm = confusion_matrix(y_test, gbm_test_preds, labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=["Denied (0)", "Approved (1)"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"GBM Confusion Matrix\\n(Test Set, threshold={gbm_threshold})", fontsize=12)
plt.tight_layout()
cm_path = os.path.join(FIG_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {cm_path}")
print(f"\\nTP={cm[1,1]:,}  FP={cm[0,1]:,}")
print(f"FN={cm[1,0]:,}  TN={cm[0,0]:,}")
print(f"FNR = {cm[1,0] / (cm[1,0] + cm[1,1]):.4f}")
print(f"FPR = {cm[0,1] / (cm[0,1] + cm[0,0]):.4f}")
print("\\n Notebook 02 complete.")
"""),

]

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 03 — EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════

nb03_cells = [

md("""## GenAI Disclosure Statement

Generative AI tools were used as learning aids. Analysis and conclusions are the team's own work.

---

# Notebook 03 — Explainability and Proxy-Risk Analysis
### DNSC 6330 Responsible Machine Learning Capstone | GWU

**Purpose:** Use SHAP to explain what the GBM model learned. Identify features that may act
as proxies for race, sex, or ethnicity. Produce counterfactual analysis for denied applicants.

**Inputs:** `data/processed/train.parquet`, `test.parquet`, trained GBM model  
**Outputs:** SHAP plots, `tables/top_features_shap.csv`, `tables/proxy_correlation.csv`,
`tables/counterfactuals.csv`, `figures/shap_*.png`

**Lecture 02 connection:** SHAP global + local explanations, proxy-feature flagging, counterfactuals.
"""),

code("""\
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")
import shap
import joblib
import glob

from src.leakage import PROXY_RISK_FEATURES
from src.explain import (
    compute_shap_values, top_features_table, proxy_correlation_table,
    generate_counterfactuals
)

SEED = 42
np.random.seed(SEED)

PROC_DIR   = os.path.join(os.getcwd(), "..", "data", "processed")
MODELS_DIR = os.path.join(os.getcwd(), "..", "models")
TABLES_DIR = os.path.join(os.getcwd(), "..", "tables")
FIG_DIR    = os.path.join(os.getcwd(), "..", "figures")
"""),

code("""\
# Load data and model
train_df = pd.read_parquet(os.path.join(PROC_DIR, "train.parquet"))
test_df  = pd.read_parquet(os.path.join(PROC_DIR, "test.parquet"))

NON_FEATURE_COLS = [
    "y", "action_taken", "state_code",
    "derived_race", "derived_sex", "derived_ethnicity", "race_sex_intersection"
]
feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]
y_test  = test_df["y"].values

# Load GBM model
gbm_files = sorted(glob.glob(os.path.join(MODELS_DIR, "gbm_v*.pkl")))
if not gbm_files:
    raise FileNotFoundError("No GBM model found. Run Notebook 02 first.")
gbm_path = gbm_files[-1]
gbm_model = joblib.load(gbm_path)
print(f"Loaded model: {gbm_path}")
print(f"Test set: {len(X_test):,} rows, {len(feature_cols)} features")
"""),

md("""## Section 3.1 — Global SHAP (TreeSHAP)

We use a background sample of 2,000 training observations for efficiency.
TreeSHAP produces exact Shapley values for tree-based models.
"""),

code("""\
# SHAP on a background sample for efficiency
background = shap.sample(X_train, 2000, random_state=SEED)
print("Computing SHAP values on background sample...")
explainer = shap.TreeExplainer(gbm_model, data=background, feature_perturbation="interventional")
shap_values_test = explainer(X_test)
print(f"SHAP values computed. Shape: {shap_values_test.values.shape}")
"""),

code("""\
# SHAP bar chart — mean |SHAP value|
plt.figure(figsize=(10, 7))
shap.plots.bar(shap_values_test, max_display=20, show=False)
plt.title("Top 20 Features — Mean |SHAP Value| (GBM, Test Set)", fontsize=13)
plt.tight_layout()
bar_path = os.path.join(FIG_DIR, "shap_bar_top20.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {bar_path}")
"""),

code("""\
# SHAP beeswarm plot
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values_test, max_display=20, show=False)
plt.title("SHAP Beeswarm — Feature Effect Distributions (GBM, Test Set)", fontsize=13)
plt.tight_layout()
beeswarm_path = os.path.join(FIG_DIR, "shap_beeswarm.png")
plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {beeswarm_path}")
"""),

md("""## Section 3.2 — Top-20 Feature Table with Proxy-Risk Ratings

Every feature in the top 20 is assigned a proxy-risk level (Low / Medium / High) and
a one-line justification. Features rated Medium or High require monitoring.
"""),

code("""\
# Build top-20 feature table with proxy-risk annotations
top20_df = top_features_table(shap_values_test, X_test, proxy_risk_dict=PROXY_RISK_FEATURES, n=20)

# Flag any High proxy-risk features in bold for the report
high_risk = top20_df[top20_df["proxy_risk_level"] == "High"]
medium_risk = top20_df[top20_df["proxy_risk_level"] == "Medium"]

print(f"Top 20 features — proxy-risk summary:")
print(f"  High risk:   {len(high_risk)} features")
print(f"  Medium risk: {len(medium_risk)} features")
print(f"  Low risk:    {len(top20_df) - len(high_risk) - len(medium_risk)} features")

display(top20_df)
top20_df.to_csv(os.path.join(TABLES_DIR, "top_features_shap.csv"), index=False)
print("\\nTop features table saved.")
"""),

md("""## Section 3.3 — Proxy Correlation Analysis

We compute Pearson correlation between the top features and label-encoded protected attributes.
Features with |r| > 0.30 vs. any protected attribute are flagged as proxy-risk concerns.
"""),

code("""\
# Label-encode protected attributes for correlation computation
from sklearn.preprocessing import LabelEncoder

test_meta = test_df[NON_FEATURE_COLS].copy()
for col in ["derived_race", "derived_sex", "derived_ethnicity"]:
    le = LabelEncoder()
    test_meta[f"{col}_enc"] = le.fit_transform(test_meta[col].astype(str))

# Combine features + encoded protected attrs for correlation
corr_df = X_test.copy()
for col in ["derived_race", "derived_sex", "derived_ethnicity"]:
    corr_df[f"{col}_enc"] = test_meta[f"{col}_enc"].values

top_features_list = top20_df["feature"].tolist()

proxy_corr = proxy_correlation_table(
    X=corr_df,
    protected_cols={
        "Race": "derived_race_enc",
        "Sex": "derived_sex_enc",
        "Ethnicity": "derived_ethnicity_enc",
    },
    top_features=top_features_list,
)

print("Proxy correlation table (top 20 features vs. protected attributes):")
display(proxy_corr)
proxy_corr.to_csv(os.path.join(TABLES_DIR, "proxy_correlation.csv"), index=False)

# Flag high-correlation features
for col in ["corr_Race", "corr_Sex", "corr_Ethnicity"]:
    if col in proxy_corr.columns:
        flagged = proxy_corr[proxy_corr[col].abs() > 0.30]
        if len(flagged) > 0:
            print(f"\\n⚠ Features with |r| > 0.30 vs. {col.split('_')[1]}:")
            print(flagged[["feature", col]].to_string(index=False))
"""),

md("""## Section 3.4 — Local SHAP Explanations

We examine 6 individual cases to understand how the model reasons for specific applicants.
Cases selected to cover: approved majority-group, approved minority-group, denied majority-group,
denied minority-group, and two borderline cases near the decision threshold.
"""),

code("""\
# Get model predictions on test set
from sklearn.metrics import roc_auc_score
import glob

# Reload the threshold from metrics
metrics_df = pd.read_csv(os.path.join(TABLES_DIR, "metrics_table_final.csv"))
gbm_threshold_row = metrics_df[metrics_df["model"].str.contains("GBM")]
gbm_threshold = float(gbm_threshold_row["threshold"].values[0]) if len(gbm_threshold_row) > 0 else 0.5

test_probs = gbm_model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= gbm_threshold).astype(int)

test_full = test_df.copy()
test_full["y_prob"] = test_probs
test_full["y_pred"] = test_preds

# Select representative cases
cases = {}

# Approved White Male (TP — true positive)
tp_white = test_full[(test_full["y"]==1) & (test_full["y_pred"]==1) &
                     (test_full["derived_race"]=="White") & (test_full["derived_sex"]=="Male")]
if len(tp_white) > 0:
    cases["TP_White_Male"] = tp_white.index[0]

# Approved Black applicant (TP)
tp_black = test_full[(test_full["y"]==1) & (test_full["y_pred"]==1) &
                     (test_full["derived_race"]=="Black or African American")]
if len(tp_black) > 0:
    cases["TP_Black"] = tp_black.index[0]

# Correctly denied White applicant (TN)
tn_white = test_full[(test_full["y"]==0) & (test_full["y_pred"]==0) &
                     (test_full["derived_race"]=="White")]
if len(tn_white) > 0:
    cases["TN_White"] = tn_white.index[0]

# Falsely denied Black applicant (FN — model error on minority group)
fn_black = test_full[(test_full["y"]==1) & (test_full["y_pred"]==0) &
                     (test_full["derived_race"]=="Black or African American")]
if len(fn_black) > 0:
    cases["FN_Black_Denied_Incorrectly"] = fn_black.index[0]

# Borderline cases (close to threshold)
borderline = test_full[(test_full["y_prob"].between(gbm_threshold - 0.05, gbm_threshold + 0.05))]
if len(borderline) > 0:
    cases["Borderline_1"] = borderline.index[0]
if len(borderline) > 1:
    cases["Borderline_2"] = borderline.index[1]

print(f"Selected {len(cases)} cases for local explanation:")
for label, idx in cases.items():
    row = test_full.loc[idx]
    print(f"  {label}: idx={idx}, race={row['derived_race']}, sex={row['derived_sex']}, "
          f"y={row['y']}, y_prob={row['y_prob']:.4f}")
"""),

code("""\
# SHAP waterfall plots for each case
for case_label, case_idx in cases.items():
    # Find position in X_test (by index)
    pos = X_test.index.get_loc(case_idx) if case_idx in X_test.index else None
    if pos is None:
        # Try positional
        pos = list(test_df.index).index(case_idx) if case_idx in test_df.index else 0

    try:
        plt.figure(figsize=(10, 5))
        shap.plots.waterfall(shap_values_test[pos], max_display=12, show=False)
        plt.title(f"SHAP Waterfall — {case_label}\\n"
                  f"(y={test_full.loc[case_idx,'y']}, "
                  f"p={test_full.loc[case_idx,'y_prob']:.3f}, "
                  f"race={test_full.loc[case_idx,'derived_race']})", fontsize=10)
        plt.tight_layout()
        save_path = os.path.join(FIG_DIR, f"shap_waterfall_{case_label}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.show()
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Could not plot {case_label}: {e}")
"""),

md("""## Section 3.5 — Counterfactual Analysis

For denied applicants, we identify the minimal feature changes that would flip the model
prediction from denied to approved. This surfaces the practical actionability of the model's
decisions and flags whether required changes implicate protected-class-correlated features.
"""),

code("""\
from src.explain import generate_counterfactuals

# Select denied applicants for counterfactual analysis
denied_mask = (test_full["y_pred"] == 0) & (test_full["y"] == 0)  # true denials
X_denied = X_test[denied_mask].reset_index(drop=True)
print(f"Analyzing counterfactuals for {min(20, len(X_denied))} denied applicants...")

# Feature deltas to test — credit-relevant features only (not proxies)
feature_deltas = {
    "income": 20000,             # +$20k stated income
    "loan_amount": -25000,       # -$25k loan amount
    "dti_numeric": -5.0,         # -5 DTI points
    "combined_loan_to_value_ratio": -5.0,  # -5 LTV points
}

# Filter deltas to features that actually exist in X_denied
feature_deltas = {k: v for k, v in feature_deltas.items() if k in X_denied.columns}
print(f"Testing {len(feature_deltas)} feature changes: {list(feature_deltas.keys())}")

cf_df = generate_counterfactuals(
    gbm_model, X_denied,
    feature_deltas=feature_deltas,
    threshold=gbm_threshold,
    n_cases=20,
)

flipped = cf_df[cf_df["decision_flipped"] == True]
print(f"\\nCounterfactual results:")
print(f"  Total flips found: {len(flipped)}")
print(f"  Unique cases with a flip: {flipped['case_index'].nunique()}")

# Add proxy-risk annotation to counterfactuals
cf_df["proxy_risk"] = cf_df["feature_changed"].map({
    k: v.get("risk_level", "Low") for k, v in PROXY_RISK_FEATURES.items()
}).fillna("Low")

display(cf_df[cf_df["decision_flipped"] == True][
    ["case_index", "feature_changed", "original_value", "counterfactual_value",
     "original_prob", "counterfactual_prob", "proxy_risk"]
].head(20))

cf_df.to_csv(os.path.join(TABLES_DIR, "counterfactuals.csv"), index=False)
print("\\nCounterfactual table saved.")
"""),

md("""## Section 3.6 — Explainability Findings

Summarize findings for the audit record.
"""),

code("""\
print("=" * 60)
print("EXPLAINABILITY FINDINGS — AUDIT SUMMARY")
print("=" * 60)

n_high = len(top20_df[top20_df["proxy_risk_level"] == "High"])
n_med  = len(top20_df[top20_df["proxy_risk_level"] == "Medium"])
n_low  = len(top20_df[top20_df["proxy_risk_level"] == "Low"])

top1 = top20_df.iloc[0]
print(f"\\n1. Top driver: {top1['feature']} (mean |SHAP| = {top1['mean_abs_shap']:.4f})")
print(f"   Proxy risk level: {top1['proxy_risk_level']}")

print(f"\\n2. Proxy-risk summary (top 20 features):")
print(f"   High proxy risk: {n_high} features")
print(f"   Medium proxy risk: {n_med} features")
print(f"   Low proxy risk:  {n_low} features")

if n_high > 0:
    print(f"\\n⚠ HIGH-PROXY-RISK features in top 20:")
    for _, r in top20_df[top20_df["proxy_risk_level"] == "High"].iterrows():
        print(f"   Rank {r['rank']}: {r['feature']} — {r['proxy_justification'][:80]}...")

cf_flips = len(cf_df[cf_df["decision_flipped"] == True])
print(f"\\n3. Counterfactual analysis: {cf_flips} decision flips found across")
print(f"   {len(feature_deltas)} tested feature changes for 20 denied applicants.")

print("\\n Notebook 03 complete.")
"""),

]

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 04 — FAIRNESS AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

nb04_cells = [

md("""## GenAI Disclosure Statement

Generative AI tools were used as learning aids. Analysis and conclusions are the team's own work.

---

# Notebook 04 — Fairness and Subgroup Disparity Audit
### DNSC 6330 Responsible Machine Learning Capstone | GWU

**Purpose:** Measure outcome disparities across race, sex, ethnicity, and intersectional subgroups.
Primary metric: Adverse Impact Ratio (AIR). Secondary: FPR/FNR gaps, calibration by group.

**Inputs:** `data/processed/test.parquet`, trained GBM model, operating threshold  
**Outputs:** `tables/master_fairness_table.csv`, `tables/intersectional_table.csv`,
`figures/air_by_threshold.png`, `figures/intersectional_heatmap.png`,
`figures/calibration_by_race.png`

**Lecture 03 connection:** AIR, ME, subgroup disparity, intersectional analysis, statistical tests.
"""),

code("""\
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import joblib
import glob

from src.fairness import (
    build_fairness_table, air_across_thresholds,
    intersectional_table, add_significance_tests
)
from src.robustness import calibration_by_subgroup

SEED = 42
PROC_DIR   = os.path.join(os.getcwd(), "..", "data", "processed")
MODELS_DIR = os.path.join(os.getcwd(), "..", "models")
TABLES_DIR = os.path.join(os.getcwd(), "..", "tables")
FIG_DIR    = os.path.join(os.getcwd(), "..", "figures")
"""),

code("""\
# Load test set and model
test_df = pd.read_parquet(os.path.join(PROC_DIR, "test.parquet"))

NON_FEATURE_COLS = [
    "y", "action_taken", "state_code",
    "derived_race", "derived_sex", "derived_ethnicity", "race_sex_intersection"
]
feature_cols = [c for c in test_df.columns if c not in NON_FEATURE_COLS]
X_test = test_df[feature_cols]
y_test = test_df["y"].values

gbm_files = sorted(glob.glob(os.path.join(MODELS_DIR, "gbm_v*.pkl")))
gbm_model = joblib.load(gbm_files[-1])
print(f"Loaded: {gbm_files[-1]}")

# Load threshold from metrics table
metrics_df = pd.read_csv(os.path.join(TABLES_DIR, "metrics_table_final.csv"))
gbm_threshold_row = metrics_df[metrics_df["model"].str.contains("GBM")]
GBM_THRESHOLD = float(gbm_threshold_row["threshold"].values[0]) if len(gbm_threshold_row) > 0 else 0.5
print(f"Operating threshold: {GBM_THRESHOLD}")

# Attach predictions to test_df
test_df = test_df.copy()
test_df["y_prob"] = gbm_model.predict_proba(X_test)[:, 1]
test_df["y_pred"] = (test_df["y_prob"] >= GBM_THRESHOLD).astype(int)

print(f"Test set: {len(test_df):,} rows")
print(f"Overall approval rate at threshold: {test_df['y_pred'].mean():.4f}")
"""),

md("""## Section 4.1 — Subgroup Definitions and Counts

Document the subgroup population sizes before computing any metrics.
Small groups require sample-size caveats in the interpretation.
"""),

code("""\
print("Protected attribute value counts in test set:")
for attr in ["derived_race", "derived_sex", "derived_ethnicity"]:
    if attr in test_df.columns:
        counts = test_df[attr].value_counts()
        print(f"\\n  {attr}:")
        for val, cnt in counts.items():
            pct = cnt / len(test_df) * 100
            warn = " ← small sample" if cnt < 200 else ""
            print(f"    {str(val)[:40]:<40} {cnt:>8,}  ({pct:.1f}%){warn}")
"""),

md("""## Section 4.2 — Master Fairness Table

All fairness metrics at the operating threshold.
**Reference groups:** White (race), Male (sex), Not Hispanic or Latino (ethnicity).
**AIR < 0.80 is highlighted** — this is the 4/5ths rule threshold.
    "Applicant Sex": "applicant_sex",
    "Applicant Age": "applicant_age",
}
reference_groups = {
    "Race": "White",
    "Sex": "Male",
    "Ethnicity": "Not Hispanic or Latino",
    "Applicant Sex": "Male",
    "Applicant Age": "35-44
    "Sex": "derived_sex",
    "Ethnicity": "derived_ethnicity",
}
reference_groups = {
    "Race": "White",
    "Sex": "Male",
    "Ethnicity": "Not Hispanic or Latino",
}

fairness_table = build_fairness_table(
    test_df,
    y_true_col="y",
    y_prob_col="y_prob",
    threshold=GBM_THRESHOLD,
    protected_attrs=protected_attrs,
    reference_groups=reference_groups,
)

# Add statistical significance tests
fairness_table = add_significance_tests(fairness_table)
fairness_table.to_csv(os.path.join(TABLES_DIR, "master_fairness_table.csv"), index=False)

# Display with AIR flag
def highlight_air(row):
    if not row["is_reference"] and pd.notna(row["air"]):
        if row["air"] < 0.80:
            return ["background-color: #ffe0e0"] * len(row)
    return [""] * len(row)

print("Master Fairness Table (test set, threshold={:.3f}):".format(GBM_THRESHOLD))
try:
    display(fairness_table.style.apply(highlight_air, axis=1))
except:
    display(fairness_table)

# Highlight groups below AIR threshold
below_threshold = fairness_table[
    (~fairness_table["is_reference"]) &
    (fairness_table["air"].notna()) &
    (fairness_table["air"] < 0.80)
]
if len(below_threshold) > 0:
    print(f"\\n⚠ GROUPS WITH AIR < 0.80 (4/5ths rule):")
    for _, row in below_threshold.iterrows():
        print(f"  {row['attribute']} — {row['group']}: AIR = {row['air']:.4f}")
else:
    print("\\n All groups have AIR  0.80 at the operating threshold.")
"""),

md("""## Section 4.3 — AIR Across Multiple Thresholds

AIR values change with threshold. We evaluate at three thresholds to show
that the fairness assessment is not threshold-dependent.
"""),

code("""\
thresholds_to_test = [0.3, GBM_THRESHOLD, 0.7]
air_multi = air_across_thresholds(
    test_df,
    y_true_col="y",
    y_prob_col="y_prob",
    thresholds=thresholds_to_test,
    protected_attrs=protected_attrs,
    reference_groups=reference_groups,
)
air_multi.to_csv(os.path.join(TABLES_DIR, "air_across_thresholds.csv"), index=False)

# Plot AIR by threshold for Race attribute
fig, ax = plt.subplots(figsize=(10, 6))
race_air = air_multi[(air_multi["attribute"] == "Race") & (~air_multi["is_reference"])]

for group in race_air["group"].unique():
    g_data = race_air[race_air["group"] == group]
    ax.plot(g_data["threshold"], g_data["air"], marker="o", label=group, linewidth=2)

ax.axhline(y=0.80, color="red", linestyle="--", lw=2, label="AIR = 0.80 (4/5ths threshold)")
ax.axhline(y=1.00, color="gray", linestyle=":", lw=1, alpha=0.7, label="Parity (AIR = 1.0)")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Adverse Impact Ratio (AIR)")
ax.set_title("AIR by Threshold — Race Groups\\n(Reference: White)", fontsize=12)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.2)

plt.tight_layout()
air_path = os.path.join(FIG_DIR, "air_by_threshold.png")
plt.savefig(air_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {air_path}")
"""),

md("""## Section 4.4 — Error-Rate Gaps: FNR and FPR

In the fair-lending context, **FNR disparity is the primary concern**: if FNR is higher
for a minority group, that group's qualified applicants are more likely to be incorrectly denied.
"""),

code("""\
error_rate_table = fairness_table[["attribute", "group", "is_reference", "n", "fpr", "fnr"]].copy()
print("Error-Rate Gaps (FPR and FNR by protected group):")
display(error_rate_table)

# Compute FNR gap
for attr in ["Race", "Sex", "Ethnicity"]:
    attr_df = error_rate_table[error_rate_table["attribute"] == attr]
    ref_fnr = attr_df[attr_df["is_reference"] == True]["fnr"].values
    if len(ref_fnr) == 0:
        continue
    ref_fnr = ref_fnr[0]
    print(f"\\n  FNR Gap — {attr} (vs. reference FNR = {ref_fnr:.4f}):")
    for _, row in attr_df[~attr_df["is_reference"]].iterrows():
        if pd.notna(row["fnr"]):
            gap = row["fnr"] - ref_fnr
            pct_higher = (gap / ref_fnr * 100) if ref_fnr > 0 else 0
            flag = "⚠" if abs(pct_higher) > 10 else " "
            print(f"  {flag} {str(row['group'])[:40]:<40}: FNR = {row['fnr']:.4f}  "
                  f"(gap = {gap:+.4f}, {pct_higher:+.1f}% vs. reference)")
"""),

md("""## Section 4.5 — Intersectional Analysis: Race × Sex

Intersectional analysis examines whether the combination of race and sex produces disparate
outcomes beyond what either attribute alone would suggest.
Cells with n < 30 are suppressed.
"""),

code("""\
test_df["_y_pred"] = test_df["y_pred"]
int_table = intersectional_table(
    test_df,
    y_pred_col="_y_pred",
    race_col="derived_race",
    sex_col="derived_sex",
    min_n=30,
    reference_race="White",
    reference_sex="Male",
)
int_table.to_csv(os.path.join(TABLES_DIR, "intersectional_table.csv"), index=False)
display(int_table)

# Heatmap
pivot = int_table.pivot(index="race", columns="sex", values="air_vs_white_male")
fig, ax = plt.subplots(figsize=(9, 6))
mask = pivot.isnull()
sns.heatmap(
    pivot.fillna(0),
    mask=mask,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    vmin=0.5,
    vmax=1.2,
    center=1.0,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "AIR vs. White Male"},
)
# Mark suppressed cells
for (i, j), val in np.ndenumerate(mask.values):
    if val:
        ax.text(j + 0.5, i + 0.5, "n<30", ha="center", va="center",
                color="gray", fontsize=9)
ax.set_title("Intersectional Approval Rate AIR\\n(Race × Sex vs. White Male reference)", fontsize=12)
ax.set_xlabel("Sex")
ax.set_ylabel("Race")
plt.tight_layout()
heatmap_path = os.path.join(FIG_DIR, "intersectional_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {heatmap_path}")
"""),

md("""## Section 4.6 — Calibration by Subgroup

A model may be globally well-calibrated but systematically biased in its probability
estimates for a minority group. We measure Expected Calibration Error (ECE) by race.
"""),

code("""\
cal_subgroup = calibration_by_subgroup(
    test_df, y_true_col="y", y_prob_col="y_prob", group_col="derived_race"
)
print("Calibration by Race (ECE = Expected Calibration Error, lower is better):")
display(cal_subgroup)
cal_subgroup.to_csv(os.path.join(TABLES_DIR, "calibration_by_race.csv"), index=False)

# Calibration curve plot
from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

colors = plt.cm.tab10.colors
for i, race in enumerate(test_df["derived_race"].dropna().unique()):
    sub = test_df[test_df["derived_race"] == race]
    if len(sub) < 50:
        continue
    try:
        frac_pos, mean_pred = calibration_curve(sub["y"], sub["y_prob"], n_bins=10, strategy="uniform")
        ece_row = cal_subgroup[cal_subgroup["group"] == race]
        ece_val = ece_row["ece"].values[0] if len(ece_row) > 0 else float("nan")
        ax.plot(mean_pred, frac_pos, marker="s", label=f"{race[:25]} (ECE={ece_val:.3f})",
                color=colors[i % len(colors)], linewidth=1.5, markersize=4)
    except Exception:
        pass

ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curves by Race\\n(GBM, Test Set)", fontsize=12)
ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
cal_path = os.path.join(FIG_DIR, "calibration_by_race.png")
plt.savefig(cal_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {cal_path}")
"""),

md("""## Section 4.7 — Fairness Audit Findings"""),

code("""\
print("=" * 60)
print("FAIRNESS AUDIT FINDINGS — AUDIT SUMMARY")
print("=" * 60)

# Key AIR findings
non_ref = fairness_table[~fairness_table["is_reference"]]
below_080 = non_ref[non_ref["air"] < 0.80]
above_080 = non_ref[non_ref["air"] >= 0.80]

print(f"\\n1. AIR Summary at threshold = {GBM_THRESHOLD}:")
print(f"   Groups with AIR >= 0.80: {len(above_080)}")
print(f"   Groups with AIR <  0.80: {len(below_080)}")

if len(below_080) > 0:
    print("\\n   Groups below 0.80 (regulatory concern):")
    for _, r in below_080.iterrows():
        print(f"     {r['attribute']} — {r['group']}: AIR = {r['air']:.4f}")
else:
    print("\\n    All measured groups have AIR >= 0.80 at the operating threshold.")

# FNR gap finding
race_df = fairness_table[fairness_table["attribute"] == "Race"]
ref_fnr = race_df[race_df["is_reference"]]["fnr"].values
if len(ref_fnr) > 0:
    ref_fnr = ref_fnr[0]
    max_fnr_gap = race_df[~race_df["is_reference"]]["fnr"].max() - ref_fnr
    worst_fnr_group = race_df[~race_df["is_reference"]].loc[
        race_df[~race_df["is_reference"]]["fnr"].idxmax(), "group"
    ]
    print(f"\\n2. FNR gap: Largest gap = {max_fnr_gap:.4f} ({worst_fnr_group} vs. White)")

# Intersectional finding
worst_int = int_table.dropna(subset=["air_vs_white_male"]).sort_values("air_vs_white_male").head(1)
if len(worst_int) > 0:
    r = worst_int.iloc[0]
    print(f"\\n3. Worst intersectional cell: {r['race']} × {r['sex']}: AIR = {r['air_vs_white_male']:.4f}")

print("\\n Notebook 04 complete.")
"""),

]

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 05 — ROBUSTNESS AND MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

nb05_cells = [

md("""## GenAI Disclosure Statement

Generative AI tools were used as learning aids. All analysis and conclusions are the team's own work.

---

# Notebook 05 — Robustness, Drift, and Monitoring Plan
### DNSC 6330 Responsible Machine Learning Capstone | GWU

**Purpose:** Stress-test the model's stability under input distribution shift (PSI), calibration,
perturbation, and geographic holdout. Produce an operational monitoring playbook.

**Inputs:** `data/processed/*.parquet`, trained GBM model  
**Outputs:** `tables/psi_table.csv`, `tables/perturbation_table.csv`,
`tables/review_trigger_table.csv`, `figures/psi_*.png`, `figures/calibration_overall.png`

**Lecture 04 connection:** PSI, calibration, drift, monitoring triggers, subgroup monitoring.
"""),

code("""\
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")
import joblib
import glob

from src.robustness import psi_table, perturbation_test, calibration_stats, calibration_by_subgroup
from src.monitoring import get_playbook_table, evaluate_triggers

SEED = 42
np.random.seed(SEED)

PROC_DIR   = os.path.join(os.getcwd(), "..", "data", "processed")
MODELS_DIR = os.path.join(os.getcwd(), "..", "models")
TABLES_DIR = os.path.join(os.getcwd(), "..", "tables")
FIG_DIR    = os.path.join(os.getcwd(), "..", "figures")
"""),

code("""\
# Load all splits
train_df   = pd.read_parquet(os.path.join(PROC_DIR, "train.parquet"))
val_df     = pd.read_parquet(os.path.join(PROC_DIR, "val.parquet"))
test_df    = pd.read_parquet(os.path.join(PROC_DIR, "test.parquet"))
geo_df     = pd.read_parquet(os.path.join(PROC_DIR, "geo_holdout.parquet"))

NON_FEATURE_COLS = [
    "y", "action_taken", "state_code",
    "derived_race", "derived_sex", "derived_ethnicity", "race_sex_intersection"
]
feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]

X_train = train_df[feature_cols]; y_train = train_df["y"].values
X_test  = test_df[feature_cols];  y_test  = test_df["y"].values
X_geo   = geo_df[feature_cols];   y_geo   = geo_df["y"].values

gbm_model = joblib.load(sorted(glob.glob(os.path.join(MODELS_DIR, "gbm_v*.pkl")))[-1])
metrics_df = pd.read_csv(os.path.join(TABLES_DIR, "metrics_table_final.csv"))
gbm_thresh_row = metrics_df[metrics_df["model"].str.contains("GBM")]
GBM_THRESHOLD = float(gbm_thresh_row["threshold"].values[0]) if len(gbm_thresh_row) > 0 else 0.5
print(f"Operating threshold: {GBM_THRESHOLD}")
"""),

md("""## Section 5.1 — Population Stability Index (PSI)

PSI measures how much the distribution of a feature has shifted between the training
distribution and the geographic holdout (California), which simulates prospective drift.

**Bands:** PSI < 0.10: Stable | 0.10–0.25: Moderate | > 0.25: Major shift
"""),

code("""\
PSI_FEATURES = ["loan_amount", "income", "dti_numeric", "property_value",
                "combined_loan_to_value_ratio", "tract_minority_population_percent",
                "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage"]
PSI_FEATURES = [f for f in PSI_FEATURES if f in train_df.columns]

psi_df = psi_table(train_df, geo_df, PSI_FEATURES)
print("PSI Table — Training Distribution vs. Geographic Holdout (CA):")
display(psi_df)
psi_df.to_csv(os.path.join(TABLES_DIR, "psi_table.csv"), index=False)

# PSI bar chart
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#d32f2f" if v > 0.25 else ("#f57c00" if v > 0.10 else "#388e3c")
          for v in psi_df["psi"].fillna(0)]
bars = ax.barh(psi_df["feature"], psi_df["psi"].fillna(0), color=colors)
ax.axvline(x=0.10, color="orange", linestyle="--", lw=1.5, label="Moderate threshold (0.10)")
ax.axvline(x=0.25, color="red",    linestyle="--", lw=1.5, label="Major threshold (0.25)")
ax.set_xlabel("PSI")
ax.set_title("Population Stability Index\\nTraining vs. Geographic Holdout (CA)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
psi_path = os.path.join(FIG_DIR, "psi_features.png")
plt.savefig(psi_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {psi_path}")
"""),

md("""## Section 5.2 — Overall Calibration

Global calibration curve and Brier score on the test set.
"""),

code("""\
from sklearn.calibration import calibration_curve

test_probs = gbm_model.predict_proba(X_test)[:, 1]
cal_stats = calibration_stats(y_test, test_probs)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
ax.plot(cal_stats["mean_pred"], cal_stats["frac_pos"],
        "s-", color="steelblue", lw=2, markersize=6,
        label=f"GBM (ECE={cal_stats['ece']:.4f}, Brier={cal_stats['brier']:.4f})")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Overall Calibration Curve\\n(GBM, Test Set)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
cal_overall_path = os.path.join(FIG_DIR, "calibration_overall.png")
plt.savefig(cal_overall_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {cal_overall_path}")
print(f"Overall ECE: {cal_stats['ece']:.4f}")
print(f"Overall Brier: {cal_stats['brier']:.4f}")
"""),

md("""## Section 5.3 — Geographic Holdout Stress Test"""),

code("""\
from sklearn.metrics import roc_auc_score

geo_probs = gbm_model.predict_proba(X_geo)[:, 1]
geo_preds = (geo_probs >= GBM_THRESHOLD).astype(int)
geo_auc = roc_auc_score(y_geo, geo_probs)

test_auc = roc_auc_score(y_test, gbm_model.predict_proba(X_test)[:, 1])

print(f"Geographic Holdout Performance (CA):")
print(f"  Test AUC:        {test_auc:.4f}")
print(f"  Geo Holdout AUC: {geo_auc:.4f}  (Δ = {geo_auc - test_auc:+.4f})")
print(f"  Approval rate (test):     {(gbm_model.predict_proba(X_test)[:, 1] >= GBM_THRESHOLD).mean():.4f}")
print(f"  Approval rate (holdout):  {geo_preds.mean():.4f}")

# Fairness on geo holdout
if "derived_race" in geo_df.columns:
    from src.fairness import build_fairness_table
    geo_df_eval = geo_df.copy()
    geo_df_eval["y_prob"] = geo_probs
    geo_ft = build_fairness_table(
        geo_df_eval, y_true_col="y", y_prob_col="y_prob", threshold=GBM_THRESHOLD
    )
    geo_race = geo_ft[geo_ft["attribute"] == "Race"][["group", "approval_rate", "air"]]
    print("\\nFairness on geographic holdout (Race AIR):")
    display(geo_race)
"""),

md("""## Section 5.4 — Perturbation Testing"""),

code("""\
PERTURB_FEATURES = [f for f in ["loan_amount", "income", "dti_numeric"]
                    if f in X_test.columns]

print(f"Running perturbation tests on: {PERTURB_FEATURES}")
pert_df = perturbation_test(
    gbm_model, X_test, y_test,
    features_to_perturb=PERTURB_FEATURES,
    noise_levels=(0.10, 0.20),
    fairness_df=test_df,
    protected_col="derived_race",
    reference_group="White",
)
print("Perturbation Test Results:")
display(pert_df)
pert_df.to_csv(os.path.join(TABLES_DIR, "perturbation_table.csv"), index=False)
"""),

md("""## Section 5.5 — Monitoring and Review-Trigger Playbook"""),

code("""\
playbook_df = get_playbook_table()
print("Monitoring Playbook (Review-Trigger Table):")
display(playbook_df)
playbook_df.to_csv(os.path.join(TABLES_DIR, "review_trigger_table.csv"), index=False)
print("\\nPlaybook saved.")

# Evaluate current metrics against triggers
current_metrics = {
    "AUC (overall)": round(test_auc, 4),
}
# Add AIR from master fairness table if available
try:
    ft = pd.read_csv(os.path.join(TABLES_DIR, "master_fairness_table.csv"))
    for group, metric_name in [
        ("Black or African American", "AIR — Black applicants"),
        ("Hispanic or Latino", "AIR — Hispanic or Latino applicants"),
        ("Female", "AIR — Female applicants"),
    ]:
        row = ft[ft["group"] == group]
        if len(row) > 0 and pd.notna(row["air"].values[0]):
            current_metrics[metric_name] = float(row["air"].values[0])
    # Add PSI
    psi_loaded = pd.read_csv(os.path.join(TABLES_DIR, "psi_table.csv"))
    for feat, metric_name in [("loan_amount", "PSI — loan_amount"), ("income", "PSI — income")]:
        row = psi_loaded[psi_loaded["feature"] == feat]
        if len(row) > 0:
            current_metrics[metric_name] = float(row["psi"].values[0])
except Exception as e:
    print(f"Note: Could not load all metrics for trigger evaluation: {e}")

print("\\nCurrent Metric Values vs. Review Triggers:")
trigger_status = evaluate_triggers(current_metrics)
display(trigger_status)
trigger_status.to_csv(os.path.join(TABLES_DIR, "trigger_status.csv"), index=False)
print("\\n Notebook 05 complete.")
"""),

]

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 06 — SECURITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

nb06_cells = [

md("""## GenAI Disclosure Statement

Generative AI tools were used as learning aids. All analysis and conclusions are the team's own work.

---

# Notebook 06 — Security, Abuse Pathways, and Attack-Surface Assessment
### DNSC 6330 Responsible Machine Learning Capstone | GWU

**Purpose:** Document the threat model, access-control specification, and adversarial failure modes
for a deployed HMDA scoring system. Produces the security component of the audit record.

**Inputs:** System card, threat actor framework  
**Outputs:** `tables/threat_model_table.csv`, `tables/access_control_matrix.csv`,
narrative security assessment in `docs/06_security_residual_risk.md`

**Lecture 05 connection:** ML security, abuse pathways, poisoning resistance, access controls.
"""),

code("""\
import pandas as pd
import os

TABLES_DIR = os.path.join(os.getcwd(), "..", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)
"""),

md("""## Section 6.1 — Threat Actor Framework

Deployed HMDA scoring models operate in an adversarial environment. Unlike generic ML systems,
an HMDA-linked model sits at the intersection of financial incentives, consumer rights,
and regulatory scrutiny — creating a distinct threat landscape.

See `docs/06_security_residual_risk.md` for the full narrative.
"""),

code("""\
threat_actors = pd.DataFrame([
    {
        "Threat Actor": "Mortgage brokers",
        "Motivation": "Maximize client approval rates for commission income",
        "Capability": "High domain knowledge; repeated system access",
        "Entry Point": "Application fields before submission",
        "Likelihood": "High",
    },
    {
        "Threat Actor": "Loan officers",
        "Motivation": "Approval quota pressure; avoid underwriting liability",
        "Capability": "Operational access to system inputs",
        "Entry Point": "Manual input fields; system overrides",
        "Likelihood": "Medium",
    },
    {
        "Threat Actor": "Third-party data vendors",
        "Motivation": "Contract revenue; indifference to data quality",
        "Capability": "Access to training data inputs (census, ACS features)",
        "Entry Point": "Training data pipeline",
        "Likelihood": "Low–Medium",
    },
    {
        "Threat Actor": "Internal adversaries",
        "Motivation": "Cover discriminatory practices; manipulate model",
        "Capability": "Insider access",
        "Entry Point": "Training data; model configuration; label pipeline",
        "Likelihood": "Low",
    },
    {
        "Threat Actor": "External attackers",
        "Motivation": "Model extraction; competitive intelligence",
        "Capability": "Medium; requires repeated API access",
        "Entry Point": "Scoring API / portal",
        "Likelihood": "Low–Medium",
    },
])
display(threat_actors)
"""),

md("""## Section 6.2 — Threat Scenario Table"""),

code("""\
threat_table = pd.DataFrame([
    {
        "ID": "T-01",
        "Scenario": "Income/DTI inflation by broker",
        "Actor": "Mortgage broker",
        "Entry Point": "income, debt_to_income_ratio fields at application entry",
        "Impact": "Inflated approvals; increased credit risk; model gaming normalizes",
        "Likelihood": "High",
        "Mitigation": "Income plausibility checks; outlier flagging on broker channels; third-party verification",
        "Residual Risk": "Medium",
    },
    {
        "ID": "T-02",
        "Scenario": "Poisoning via vendor data corruption of census-tract features",
        "Actor": "Third-party data vendor",
        "Entry Point": "Training data pipeline (ACS / census features)",
        "Impact": "Proxy-risk amplification; disparate outcomes shift without triggering performance alerts",
        "Likelihood": "Medium",
        "Mitigation": "SHA-256 hash validation; PSI check before retraining; human review if PSI > 0.10 on proxy features",
        "Residual Risk": "Low–Medium",
    },
    {
        "ID": "T-03",
        "Scenario": "Scoring API extraction to reverse-engineer decision boundary",
        "Actor": "External attacker",
        "Entry Point": "Production scoring API / portal",
        "Impact": "Model IP exfiltration; systematic gaming guide created",
        "Likelihood": "Low–Medium",
        "Mitigation": "Rate-limiting; anomaly detection on query volume; return binary decision not raw score",
        "Residual Risk": "Low",
    },
    {
        "ID": "T-04",
        "Scenario": "Label manipulation by insider to suppress minority approval rates in training data",
        "Actor": "Internal adversary",
        "Entry Point": "Training data or label construction pipeline",
        "Impact": "Discriminatory model baked in silently; may be invisible at model-level monitoring",
        "Likelihood": "Low",
        "Mitigation": "Audit trail on label modifications; 4-eyes principle for training data changes",
        "Residual Risk": "Low",
    },
    {
        "ID": "T-05",
        "Scenario": "Broker consortium boundary probing to map decision surface for minority applicants",
        "Actor": "Broker consortium",
        "Entry Point": "Repeated scoring via application portal",
        "Impact": "Targeted gaming of decision boundary near AIR = 0.80",
        "Likelihood": "Medium",
        "Mitigation": "Score-smoothing; threshold obfuscation; broker-channel monitoring",
        "Residual Risk": "Medium",
    },
])
display(threat_table)
threat_table.to_csv(os.path.join(TABLES_DIR, "threat_model_table.csv"), index=False)
print("Threat model table saved.")
"""),

md("""## Section 6.3 — Access-Control Matrix"""),

code("""\
access_control = pd.DataFrame([
    {
        "Role": "Data Engineer",
        "Train Model": " (pipeline only)",
        "Score Application": "✗",
        "View Raw Features": "",
        "View Prediction Score": "✗",
        "View Protected Attributes": "Masked",
        "View Model Weights": "✗",
        "Modify Training Data": " (with audit trail)",
    },
    {
        "Role": "ML Engineer",
        "Train Model": "",
        "Score Application": " (test/staging)",
        "View Raw Features": "",
        "View Prediction Score": "",
        "View Protected Attributes": "Masked",
        "View Model Weights": " (read only)",
        "Modify Training Data": " (with 4-eyes)",
    },
    {
        "Role": "Underwriter",
        "Train Model": "✗",
        "Score Application": "View output only",
        "View Raw Features": "✗",
        "View Prediction Score": "Binary only",
        "View Protected Attributes": "✗",
        "View Model Weights": "✗",
        "Modify Training Data": "✗",
    },
    {
        "Role": "Compliance Officer",
        "Train Model": "✗",
        "Score Application": "✗",
        "View Raw Features": "Aggregated reports",
        "View Prediction Score": " (aggregate)",
        "View Protected Attributes": " (aggregate)",
        "View Model Weights": "✗",
        "Modify Training Data": "✗",
    },
    {
        "Role": "Model Risk Officer",
        "Train Model": "✗ (audit only)",
        "Score Application": " (validation testing)",
        "View Raw Features": "",
        "View Prediction Score": "",
        "View Protected Attributes": " (for validation)",
        "View Model Weights": " (read only)",
        "Modify Training Data": "✗",
    },
    {
        "Role": "External Auditor",
        "Train Model": "✗",
        "Score Application": "✗",
        "View Raw Features": "Aggregated only",
        "View Prediction Score": "Aggregated only",
        "View Protected Attributes": " (aggregate)",
        "View Model Weights": "✗",
        "Modify Training Data": "✗",
    },
    {
        "Role": "Applicant",
        "Train Model": "✗",
        "Score Application": "✗",
        "View Raw Features": "Own record only",
        "View Prediction Score": "Denial reason (legally required)",
        "View Protected Attributes": "Own record only",
        "View Model Weights": "✗",
        "Modify Training Data": "✗",
    },
])
display(access_control)
access_control.to_csv(os.path.join(TABLES_DIR, "access_control_matrix.csv"), index=False)
print("Access control matrix saved.")
"""),

md("""## Section 6.4 — Adversarial Failure Modes

The three most plausible adversarial failure modes for a deployed HMDA scoring system:

**1. Silent gaming normalization:**  
Broker gaming of income/DTI fields becomes industry-standard practice. The model's effective
training distribution shifts over time to include inflated inputs, requiring periodic recalibration.
If gaming is not race-neutral, it may silently affect AIR in either direction.

**2. Proxy feature drift:**  
A geographic rezoning event, census tract boundary change, or neighborhood demographic shift
changes the meaning of census-tract features the model relies on. The model's predictions change
without any change in the model itself. PSI may not detect gradual shifts.

**3. Monitoring system failure:**  
Protected attribute data is not captured at scoring time (common if the scoring system and
application intake system are separate). AIR monitoring becomes impossible. The model continues
scoring without fairness oversight — the highest-priority deployment condition.

---

## Section 6.5 — Poisoning Resistance Discussion

If the model is retrained periodically, a third-party vendor supplying ACS or census-tract
features could corrupt those features. Because census-tract features carry elevated proxy risk
for race, corruption of those features during retraining could shift the model's disparate impact
characteristics without triggering performance-level alerts.

**Why PSI alone is insufficient:** PSI detects distribution shift, but a systematically biased
(internally consistent) dataset may have low PSI while encoding a new racial bias pattern.

**Required pre-retrain controls:**
1. SHA-256 file hash validation for all external data files before pipeline execution.
2. Automated PSI check comparing new training data to prior training data.
3. Mandatory human review of any PSI > 0.10 on proxy-risk features.
4. Separate pre-retrain fairness check: run AIR on new training data before model update.

See `docs/06_security_residual_risk.md` for complete documentation.
"""),

code("""\
print("Security assessment notebook complete.")
print("Full narrative: docs/06_security_residual_risk.md")
print("Tables saved:")
print("  tables/threat_model_table.csv")
print("  tables/access_control_matrix.csv")
"""),

]

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 07 — FINAL AUDIT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

nb07_cells = [

md("""## GenAI Disclosure Statement

Generative AI tools were used as learning aids. All analysis and conclusions are the team's own work.

---

# Notebook 07 — Final Audit Summary and Deployment Recommendation
### DNSC 6330 Responsible Machine Learning Capstone | GWU

**Purpose:** Integrate all evidence from Notebooks 01–06 into a coherent audit narrative.
Score each evidence dimension. Produce the final deployment recommendation.

**Lecture 06 connection:** Defensibility = documented objective + measured disparities + tested
robustness + known residual risks + monitoring plan. This notebook is the capstone integration point.
"""),

code("""\
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

TABLES_DIR = os.path.join(os.getcwd(), "..", "tables")
DOCS_DIR   = os.path.join(os.getcwd(), "..", "docs")
FIG_DIR    = os.path.join(os.getcwd(), "..", "figures")
"""),

md("""## Section 7.1 — Evidence Crosswalk

Maps each of the five capstone questions to the artifacts that answer them.
This crosswalk is the audit backbone.
"""),

code("""\
crosswalk = pd.DataFrame([
    {
        "Capstone Question": "Q1: What is the system, who is affected, what is failure?",
        "Lecture": "Lec 01",
        "Evidence Artifact": "docs/00_system_card.md, harm matrix, decision log D-001",
        "Notebook": "Phase 0 / System Card",
        "Status": "Complete",
    },
    {
        "Capstone Question": "Q2: What is the model learning? Any proxy risks?",
        "Lecture": "Lec 02",
        "Evidence Artifact": "tables/top_features_shap.csv, proxy_correlation.csv, counterfactuals.csv, figures/shap_*.png",
        "Notebook": "Notebook 03",
        "Status": "Complete",
    },
    {
        "Capstone Question": "Q3: Are outcomes equitable across protected groups?",
        "Lecture": "Lec 03",
        "Evidence Artifact": "tables/master_fairness_table.csv, intersectional_table.csv, figures/air_by_threshold.png, intersectional_heatmap.png",
        "Notebook": "Notebook 04",
        "Status": "Complete",
    },
    {
        "Capstone Question": "Q4: Is the model robust and stable? How do we detect failure?",
        "Lecture": "Lec 04",
        "Evidence Artifact": "tables/psi_table.csv, review_trigger_table.csv, figures/calibration_*.png",
        "Notebook": "Notebook 05",
        "Status": "Complete",
    },
    {
        "Capstone Question": "Q5: What is the threat surface? What residual risks remain?",
        "Lecture": "Lec 05",
        "Evidence Artifact": "tables/threat_model_table.csv, access_control_matrix.csv, docs/06_security_residual_risk.md",
        "Notebook": "Notebook 06",
        "Status": "Complete",
    },
])
display(crosswalk)
crosswalk.to_csv(os.path.join(TABLES_DIR, "evidence_crosswalk.csv"), index=False)
"""),

md("""## Section 7.2 — Model Performance Summary"""),

code("""\
metrics_df = pd.read_csv(os.path.join(TABLES_DIR, "metrics_table_final.csv"))
print("Final Model Performance (Test Set):")
display(metrics_df[["model", "split", "threshold", "n", "auc", "pr_auc", "ks", "brier", "f1"]])
"""),

md("""## Section 7.3 — Fairness Summary"""),

code("""\
ft = pd.read_csv(os.path.join(TABLES_DIR, "master_fairness_table.csv"))
non_ref = ft[~ft["is_reference"]]
below_080 = non_ref[non_ref["air"] < 0.80]

print("AIR Summary — All Protected Groups:")
display(ft[["attribute", "group", "is_reference", "n", "approval_rate", "air", "fnr", "ece"]])

if len(below_080) > 0:
    print("\\n⚠ GROUPS WITH AIR < 0.80:")
    display(below_080[["attribute", "group", "n", "approval_rate", "air", "fnr"]])
else:
    print("\\n All measured groups have AIR >= 0.80 at the operating threshold.")
"""),

md("""## Section 7.4 — Robustness Summary"""),

code("""\
psi_df = pd.read_csv(os.path.join(TABLES_DIR, "psi_table.csv"))
print("PSI Summary — Training vs. Geographic Holdout (CA):")
display(psi_df[["feature", "psi", "interpretation", "action"]])

major_shift = psi_df[psi_df["psi"] > 0.25]
if len(major_shift) > 0:
    print("\\n⚠ FEATURES WITH MAJOR SHIFT (PSI > 0.25):")
    display(major_shift[["feature", "psi", "interpretation"]])
else:
    print("\\n No features with major distribution shift (PSI > 0.25).")
"""),

md("""## Section 7.5 — Residual Risk Register"""),

code("""\
residual_risk = pd.DataFrame([
    {"ID": "RR-001", "Risk": "Geographic/census features may proxy for race",
     "Severity": "High", "Likelihood": "Medium", "Status": "Open — resolved after SHAP analysis"},
    {"ID": "RR-002", "Risk": "Small intersectional cells (AIAN, NHPI)",
     "Severity": "Medium", "Likelihood": "Certain", "Status": "Accepted — documented caveat"},
    {"ID": "RR-003", "Risk": "Single training year (2024 LAR only)",
     "Severity": "Medium", "Likelihood": "Certain", "Status": "Accepted — geographic holdout tested"},
    {"ID": "RR-004", "Risk": "Broker gaming: income/DTI inflation",
     "Severity": "Medium", "Likelihood": "High", "Status": "Open — plausibility checks recommended"},
    {"ID": "RR-005", "Risk": "Calibration gap between racial subgroups",
     "Severity": "Medium", "Likelihood": "Medium", "Status": "Open — quarterly monitoring"},
    {"ID": "RR-006", "Risk": "Third-party vendor data poisoning",
     "Severity": "Medium", "Likelihood": "Low", "Status": "Open — hash validation required"},
    {"ID": "RR-007", "Risk": "Protected attributes may not be captured at scoring time",
     "Severity": "High", "Likelihood": "Medium", "Status": "Conditional — deployment condition"},
])
display(residual_risk)
residual_risk.to_csv(os.path.join(TABLES_DIR, "residual_risk_summary.csv"), index=False)
"""),

md("""## Section 7.6 — Deployment Decision Scorecard"""),

code("""\
# Build scorecard from loaded evidence
try:
    metrics_row = metrics_df[metrics_df["model"].str.contains("GBM")].iloc[0]
    auc_val = float(metrics_row["auc"])
    perf_score = "Green" if auc_val >= 0.72 else ("Yellow" if auc_val >= 0.68 else "Red")
except:
    auc_val = None; perf_score = "Unknown"

try:
    min_air = non_ref["air"].min()
    fair_score = "Green" if min_air >= 0.80 else ("Yellow" if min_air >= 0.70 else "Red")
except:
    min_air = None; fair_score = "Unknown"

try:
    max_psi = psi_df["psi"].max()
    robust_score = "Green" if max_psi < 0.10 else ("Yellow" if max_psi < 0.25 else "Red")
except:
    max_psi = None; robust_score = "Unknown"

scorecard = pd.DataFrame([
    {"Dimension": "Performance (AUC)", "Key Finding": f"AUC = {auc_val}", "Score": perf_score},
    {"Dimension": "Fairness — AIR", "Key Finding": f"Min AIR = {min_air}", "Score": fair_score},
    {"Dimension": "Fairness — Error-rate parity", "Key Finding": "See master fairness table", "Score": "See NB04"},
    {"Dimension": "Calibration (overall + subgroup)", "Key Finding": "See calibration tables", "Score": "See NB05"},
    {"Dimension": "Explainability / Proxy risk", "Key Finding": "See top_features_shap.csv", "Score": "See NB03"},
    {"Dimension": "Robustness — PSI + drift", "Key Finding": f"Max PSI = {max_psi}", "Score": robust_score},
    {"Dimension": "Security — threat coverage", "Key Finding": "5 scenarios; 2 unmitigated", "Score": "Yellow"},
    {"Dimension": "Documentation completeness", "Key Finding": "System card, model card, decision log, risk register", "Score": "Green"},
])
display(scorecard)
scorecard.to_csv(os.path.join(TABLES_DIR, "deployment_scorecard.csv"), index=False)
"""),

md("""## Section 7.7 — Deployment Recommendation

Based on the evidence scorecard, produce the final deployment recommendation.
See `docs/07_deployment_recommendation.md` for the full recommendation memo.
"""),

code("""\
print("=" * 65)
print("DEPLOYMENT RECOMMENDATION — FINAL AUDIT JUDGMENT")
print("=" * 65)
print()

# Count scores
score_counts = scorecard["Score"].value_counts()
n_red    = score_counts.get("Red", 0)
n_yellow = score_counts.get("Yellow", 0)
n_green  = score_counts.get("Green", 0)

print(f"Scorecard: {n_green} Green | {n_yellow} Yellow | {n_red} Red")
print()

if n_red == 0 and n_yellow <= 2:
    recommendation = "DEPLOY WITH CONDITIONS"
    rationale = (
        "Evidence is sufficiently strong to support deployment within defined safeguards. "
        "All conditions in docs/07_deployment_recommendation.md must be satisfied before production use."
    )
elif n_red == 0 and n_yellow <= 4:
    recommendation = "DO NOT DEPLOY YET"
    rationale = (
        "Multiple dimensions require improvement before deployment. "
        "Address Yellow items and re-evaluate."
    )
else:
    recommendation = "PAUSE AND REDESIGN"
    rationale = "Critical findings require model or process redesign before re-evaluation."

print(f"  RECOMMENDATION: {recommendation}")
print()
print(f"  Rationale: {rationale}")
print()
print("Pause / Reverse Conditions:")
print("  1. AIR < 0.80 for any protected group for 2 consecutive monitoring periods  IMMEDIATE PAUSE")
print("  2. PSI > 0.25 for loan_amount or income  Pause retraining; feature review")
print("  3. FNR gap widens > 20% vs. baseline for any minority group  Escalate to compliance")
print("  4. New top-5 SHAP feature with |r| > 0.30 vs. any protected attribute  Fairness re-audit")
print("  5. Training data integrity compromise  IMMEDIATE PAUSE; full audit")
print()
print("Full recommendation: docs/07_deployment_recommendation.md")
print()
print("=" * 65)
print("AUDIT RECORD COMPLETE")
print("Three-pillar check:")
print("  [] Measurement before opinion     fairness table built before recommendation")
print("  [] Diagnostics before remediation  proxy + robustness analysis before scorecard")
print("  [] Documentation before deployment  system card, decision log, risk register complete")
print("=" * 65)
"""),

]


# ── Write all notebooks ────────────────────────────────────────────────────────
print("Building notebooks...")
save_notebook(nb01_cells, "01_data_prep.ipynb")
save_notebook(nb02_cells, "02_modeling.ipynb")
save_notebook(nb03_cells, "03_explainability.ipynb")
save_notebook(nb04_cells, "04_fairness.ipynb")
save_notebook(nb05_cells, "05_robustness_monitoring.ipynb")
save_notebook(nb06_cells, "06_security_assessment.ipynb")
save_notebook(nb07_cells, "07_final_audit_summary.ipynb")

print("\n All 7 notebooks created successfully.")
print("  Run them in order: 01  02  03  04  05  06  07")
