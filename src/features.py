"""
features.py — Feature engineering pipeline for HMDA capstone.

Transforms raw HMDA LAR columns into a model-ready feature matrix.
The module keeps protected attributes separate from model inputs and
defaults to application-time features only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.leakage import POST_DECISION_FEATURES

# ── Feature registry ────────────────────────────────────────────────────────
# The requested feature set from the project brief is recorded here so the
# schema is explicit in one place. The default model matrix excludes direct
# protected attributes and post-decision leakage features.
REQUESTED_RESPONSIBLE_FEATURE_SET = [
    "income",
    "debt_to_income_ratio",
    "combined_loan_to_value_ratio",
    "loan_amount",
    "property_value",
    "interest_rate",
    "rate_spread",
    "loan_term",
    "total_loan_costs",
    "origination_charges",
    "discount_points",
    "lender_credits",
    "loan_type",
    "loan_purpose",
    "lien_status",
    "occupancy_type",
    "total_units",
    "construction_method",
    "negative_amortization",
    "interest_only_payment",
    "balloon_payment",
    "other_nonamortizing_features",
    "prepayment_penalty_term",
    "derived_ethnicity",
    "derived_race",
    "derived_sex",
]

DIRECT_PROTECTED_ATTRS = [
    "derived_ethnicity",
    "derived_race",
    "derived_sex",
    "applicant_sex",
    "applicant_age",
]

# Backward-compatible alias used by notebooks and tests.
PROTECTED_ATTRS = DIRECT_PROTECTED_ATTRS

# Application-time feature groups used by default for model training.
APPLICATION_TIME_NUM_FEATURES = [
    "income",
    "debt_to_income_ratio",
    "combined_loan_to_value_ratio",
    "loan_amount",
    "property_value",
    "loan_term",
]

APPLICATION_TIME_CAT_FEATURES = [
    "loan_type",
    "loan_purpose",
    "lien_status",
    "occupancy_type",
    "total_units",
    "construction_method",
    "negative_amortization",
    "interest_only_payment",
    "balloon_payment",
    "other_nonamortizing_features",
]

# Full requested feature groups, kept for documentation and optional use.
REQUESTED_NUM_FEATURES = APPLICATION_TIME_NUM_FEATURES + [
    "interest_rate",
    "rate_spread",
    "total_loan_costs",
    "origination_charges",
    "discount_points",
    "lender_credits",
    "prepayment_penalty_term",
]

REQUESTED_CAT_FEATURES = APPLICATION_TIME_CAT_FEATURES

# Default feature sets used by build_feature_matrix().
CAT_FEATURES = APPLICATION_TIME_CAT_FEATURES
NUM_FEATURES = APPLICATION_TIME_NUM_FEATURES


# ── DTI bucket mapping ──────────────────────────────────────────────────────
# HMDA reports debt_to_income_ratio as string buckets (e.g. "20%-<30%").
# We map to the midpoint of each bucket for use as a numeric feature.
DTI_MIDPOINT_MAP = {
    "<20%": 10.0,
    "20%-<30%": 25.0,
    "30%-<36%": 33.0,
    "36": 36.0,
    "37": 37.0,
    "38": 38.0,
    "39": 39.0,
    "40": 40.0,
    "41": 41.0,
    "42": 42.0,
    "43": 43.0,
    "44": 44.0,
    "45": 45.0,
    "46": 46.0,
    "47": 47.0,
    "48": 48.0,
    "49": 49.0,
    "50%-<60%": 55.0,
    "60%": 60.0,
    ">60%": 65.0,
    "Exempt": np.nan,
    "NA": np.nan,
}


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def clean_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Convert numeric columns to float, replacing non-numeric strings with NaN.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def encode_dti(df: pd.DataFrame, col: str = "debt_to_income_ratio") -> pd.DataFrame:
    """Map DTI string buckets to numeric midpoints in place."""
    df = df.copy()
    if col in df.columns:
        df[col] = df[col].astype(str).map(DTI_MIDPOINT_MAP)
    return df


def cap_outliers(df: pd.DataFrame, cols: list, upper_pct: float = 0.999) -> pd.DataFrame:
    """
    Cap extreme outliers at the upper_pct percentile to reduce noise from
    data-entry errors (e.g., loan_amount = 9999999).
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            cap = df[col].quantile(upper_pct)
            df[col] = df[col].clip(upper=cap)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    cat_features: list = None,
    num_features: list = None,
    drop_rare_cats: bool = True,
    rare_threshold: int = 50,
    include_post_decision_features: bool = False,
) -> pd.DataFrame:
    """
    Build a model-ready feature matrix from a cleaned HMDA DataFrame.

    By default, the matrix excludes direct protected attributes and
    post-decision features. Set include_post_decision_features=True only for
    audit experiments, not for baseline training.
    """
    df = df.copy()
    df = encode_dti(df)

    if cat_features is None:
        cat_features = REQUESTED_CAT_FEATURES if include_post_decision_features else CAT_FEATURES
    if num_features is None:
        num_features = REQUESTED_NUM_FEATURES if include_post_decision_features else NUM_FEATURES

    cat_features = [c for c in _dedupe_preserve_order(cat_features) if c not in DIRECT_PROTECTED_ATTRS]
    num_features = [c for c in _dedupe_preserve_order(num_features) if c not in DIRECT_PROTECTED_ATTRS]

    if not include_post_decision_features:
        cat_features = [c for c in cat_features if c not in POST_DECISION_FEATURES]
        num_features = [c for c in num_features if c not in POST_DECISION_FEATURES]

    df = clean_numeric(df, num_features)
    cap_cols = [
        c for c in ["loan_amount", "income", "property_value", "combined_loan_to_value_ratio"]
        if c in df.columns
    ]
    df = cap_outliers(df, cap_cols)

    numeric_cols = [c for c in num_features if c in df.columns]
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    cat_dummies = []
    for col in cat_features:
        if col not in df.columns:
            continue
        series = df[col].fillna("Missing").astype(str)
        if drop_rare_cats:
            vc = series.value_counts(dropna=False)
            rare = vc[vc < rare_threshold].index
            series = series.replace(list(rare), "Other")
        dummies = pd.get_dummies(series, prefix=col, drop_first=True, dtype=int)
        cat_dummies.append(dummies)

    feature_df = df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df.index)
    if cat_dummies:
        feature_df = pd.concat([feature_df] + cat_dummies, axis=1)

    return feature_df
