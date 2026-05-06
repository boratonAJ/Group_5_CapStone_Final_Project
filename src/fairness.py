"""
fairness.py — Subgroup fairness metrics for HMDA capstone.

Implements:
    - Adverse Impact Ratio (AIR)
    - False Positive Rate (FPR) and False Negative Rate (FNR) per subgroup
    - Expected Calibration Error (ECE) per subgroup
    - Master fairness table builder
    - Intersectional analysis (race × sex)
    - AIR sensitivity across thresholds

Reference groups follow HMDA fair-lending examination conventions:
    Race:      White
    Sex:       Male
    Ethnicity: Not Hispanic or Latino
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.calibration import calibration_curve


# ── Core metric functions ────────────────────────────────────────────────────

def adverse_impact_ratio(rate_protected: float, rate_reference: float) -> float:
    """AIR = approval_rate_protected / approval_rate_reference."""
    if rate_reference == 0 or np.isnan(rate_reference):
        return np.nan
    return round(rate_protected / rate_reference, 4)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) using uniform binning.
    Measures average absolute difference between predicted probabilities
    and observed frequencies.
    """
    if len(y_true) < 20:
        return np.nan
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        return float(round(np.mean(np.abs(frac_pos - mean_pred)), 4))
    except Exception:
        return np.nan


def subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    group_label: str,
) -> dict:
    """
    Compute all fairness-relevant metrics for a single subgroup.

    Returns a dict with keys: group, n, approval_rate, fpr, fnr,
    precision, recall, ece.
    """
    n = len(y_true)
    if n == 0:
        return None

    # Confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        tn = fp = fn = tp = 0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    app_rate = float(y_pred.mean())
    ece = expected_calibration_error(y_true, y_prob)

    return {
        "group": group_label,
        "n": int(n),
        "approval_rate": round(app_rate, 4),
        "fpr": round(fpr, 4) if not np.isnan(fpr) else np.nan,
        "fnr": round(fnr, 4) if not np.isnan(fnr) else np.nan,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "ece": ece,
    }


# ── Master fairness table ────────────────────────────────────────────────────

REFERENCE_GROUPS = {
    "derived_race": "White",
    "derived_sex": "Male",
    "derived_ethnicity": "Not Hispanic or Latino",
    "applicant_sex": "Male",
    "applicant_age": "35-44",
}


def build_fairness_table(
    df: pd.DataFrame,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    threshold: float = 0.5,
    protected_attrs: dict = None,
    reference_groups: dict = None,
) -> pd.DataFrame:
    """
    Build the master fairness table for all protected attributes.

    Parameters
    ----------
    df               : DataFrame with predictions and protected attributes
    y_true_col       : column name for ground truth
    y_prob_col       : column name for predicted probabilities
    threshold        : operating threshold for label assignment
    protected_attrs  : dict {attr_name: column_name} — defaults to HMDA standard
    reference_groups : dict {attr_name: reference_group_value}

    Returns
    -------
    DataFrame: master fairness table, one row per subgroup
    """
    if protected_attrs is None:
        protected_attrs = {
            "Race": "derived_race",
            "Sex": "derived_sex",
            "Ethnicity": "derived_ethnicity",
            "Applicant Sex": "applicant_sex",
            "Applicant Age": "applicant_age",
        }
    if reference_groups is None:
        reference_groups = {
            "Race": "White",
            "Sex": "Male",
            "Ethnicity": "Not Hispanic or Latino",
            "Applicant Sex": "Male",
            "Applicant Age": "35-44",
        }

    df = df.copy()
    df["_y_pred"] = (df[y_prob_col] >= threshold).astype(int)

    rows = []
    for attr_name, col in protected_attrs.items():
        if col not in df.columns:
            continue
        ref_group = reference_groups.get(attr_name, None)
        ref_mask = df[col] == ref_group
        ref_rate = df.loc[ref_mask, "_y_pred"].mean() if ref_mask.sum() > 0 else np.nan

        for group_val in sorted(df[col].dropna().unique()):
            mask = df[col] == group_val
            g = df[mask]
            row = subgroup_metrics(
                g[y_true_col].values,
                g[y_prob_col].values,
                g["_y_pred"].values,
                group_label=str(group_val),
            )
            if row is None:
                continue
            row["attribute"] = attr_name
            row["is_reference"] = (group_val == ref_group)
            row["air"] = (
                1.0 if group_val == ref_group
                else adverse_impact_ratio(row["approval_rate"], ref_rate)
            )
            row["threshold"] = threshold
            rows.append(row)

    cols = [
        "attribute", "group", "is_reference", "n", "threshold",
        "approval_rate", "air", "fpr", "fnr", "precision", "recall", "ece",
    ]
    result = pd.DataFrame(rows)
    return result[[c for c in cols if c in result.columns]]


def air_across_thresholds(
    df: pd.DataFrame,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    thresholds: list = None,
    protected_attrs: dict = None,
    reference_groups: dict = None,
) -> pd.DataFrame:
    """
    Compute the master fairness table at multiple thresholds.
    Returns a concatenated DataFrame with a 'threshold' column.
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    all_tables = []
    for t in thresholds:
        tbl = build_fairness_table(
            df, y_true_col, y_prob_col, threshold=t,
            protected_attrs=protected_attrs,
            reference_groups=reference_groups,
        )
        all_tables.append(tbl)
    return pd.concat(all_tables, ignore_index=True)


# ── Intersectional analysis ──────────────────────────────────────────────────

def intersectional_table(
    df: pd.DataFrame,
    y_pred_col: str = "_y_pred",
    race_col: str = "derived_race",
    sex_col: str = "derived_sex",
    min_n: int = 30,
    reference_race: str = "White",
    reference_sex: str = "Male",
) -> pd.DataFrame:
    """
    Build race × sex intersectional approval-rate table with AIR
    vs. White Male reference cell.

    Cells with n < min_n are suppressed (approval_rate set to NaN).
    """
    df = df.copy()

    groups = (
        df.groupby([race_col, sex_col])
        .agg(n=(y_pred_col, "count"), approval_rate=(y_pred_col, "mean"))
        .reset_index()
    )
    groups.columns = ["race", "sex", "n", "approval_rate"]

    ref_row = groups[(groups["race"] == reference_race) & (groups["sex"] == reference_sex)]
    ref_rate = ref_row["approval_rate"].values[0] if len(ref_row) > 0 else np.nan

    groups["air_vs_white_male"] = groups["approval_rate"].apply(
        lambda r: adverse_impact_ratio(r, ref_rate)
    )
    groups["suppressed"] = groups["n"] < min_n
    groups.loc[groups["suppressed"], ["approval_rate", "air_vs_white_male"]] = np.nan
    groups["note"] = groups["suppressed"].map({True: f"n < {min_n} — suppressed", False: ""})
    return groups


# ── Statistical significance ─────────────────────────────────────────────────

def two_proportion_z_test(n1: int, p1: float, n2: int, p2: float) -> tuple:
    """
    Two-proportion z-test for difference in rates (one-tailed, protected < reference).
    Returns (z_statistic, p_value).
    """
    from scipy import stats
    count1 = int(p1 * n1)
    count2 = int(p2 * n2)
    p_pool = (count1 + count2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return np.nan, np.nan
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    p_val = stats.norm.cdf(z)  # one-tailed: P(z < observed)
    return round(z, 4), round(p_val, 4)


def add_significance_tests(
    fairness_df: pd.DataFrame,
    ref_approval_rate_col: str = "approval_rate",
) -> pd.DataFrame:
    """
    Add z-test p-values to a fairness table by comparing each group's
    approval rate to its reference group.
    """
    df = fairness_df.copy()
    df["z_stat"] = np.nan
    df["p_value"] = np.nan

    for attr in df["attribute"].unique():
        attr_df = df[df["attribute"] == attr]
        ref_rows = attr_df[attr_df["is_reference"] == True]
        if len(ref_rows) == 0:
            continue
        ref_row = ref_rows.iloc[0]
        ref_n = ref_row["n"]
        ref_rate = ref_row[ref_approval_rate_col]

        for idx, row in attr_df.iterrows():
            if row["is_reference"]:
                continue
            z, p = two_proportion_z_test(
                n1=row["n"], p1=row[ref_approval_rate_col],
                n2=ref_n, p2=ref_rate,
            )
            df.at[idx, "z_stat"] = z
            df.at[idx, "p_value"] = p

    return df
