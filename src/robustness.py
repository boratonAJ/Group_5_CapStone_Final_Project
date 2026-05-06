"""
robustness.py — PSI, calibration, and perturbation testing for HMDA capstone.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve


# ── Population Stability Index ────────────────────────────────────────────────

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI).

    Interpretation bands:
        PSI < 0.10  : Stable — no action required
        0.10–0.25   : Moderate shift — increase monitoring frequency
        PSI > 0.25  : Major shift — model review required before continued deployment

    Uses percentile-based binning derived from the expected (training) distribution.
    """
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return np.nan

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual, bins=breakpoints)[0]

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Avoid log(0)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 4)


def interpret_psi(psi: float) -> str:
    if np.isnan(psi):
        return "Cannot compute"
    if psi < 0.10:
        return "Stable"
    elif psi < 0.25:
        return "Moderate shift — monitor"
    else:
        return "Major shift — review required"


def psi_table(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    features: list,
) -> pd.DataFrame:
    """
    Compute PSI for a list of features between two DataFrames.
    Returns a summary table with PSI, interpretation, and recommended action.
    """
    rows = []
    for feat in features:
        if feat not in train_df.columns or feat not in holdout_df.columns:
            rows.append({
                "feature": feat, "psi": np.nan,
                "interpretation": "Feature missing",
                "action": "Investigate",
                "n_train": 0, "n_holdout": 0,
            })
            continue
        train_vals = pd.to_numeric(train_df[feat], errors="coerce").dropna().values
        holdout_vals = pd.to_numeric(holdout_df[feat], errors="coerce").dropna().values
        psi = compute_psi(train_vals, holdout_vals)
        interp = interpret_psi(psi)
        action = (
            "None" if psi < 0.10
            else ("Increase monitoring frequency" if psi < 0.25 else "Model review required")
        )
        rows.append({
            "feature": feat,
            "psi": psi,
            "interpretation": interp,
            "action": action,
            "n_train": len(train_vals),
            "n_holdout": len(holdout_vals),
        })
    return pd.DataFrame(rows)


# ── Calibration ───────────────────────────────────────────────────────────────

def calibration_stats(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """
    Compute calibration statistics: ECE, Brier score, and curve data.
    """
    ece = np.nan
    brier = np.nan
    frac_pos = mean_pred = np.array([])

    if len(y_true) >= 20:
        try:
            frac_pos, mean_pred = calibration_curve(
                y_true, y_prob, n_bins=n_bins, strategy="uniform"
            )
            ece = float(round(np.mean(np.abs(frac_pos - mean_pred)), 4))
            brier = float(round(brier_score_loss(y_true, y_prob), 4))
        except Exception:
            pass

    return {
        "ece": ece,
        "brier": brier,
        "frac_pos": frac_pos,
        "mean_pred": mean_pred,
    }


def calibration_by_subgroup(
    df: pd.DataFrame,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    group_col: str = "derived_race",
) -> pd.DataFrame:
    """
    Compute ECE and Brier score for each subgroup of a categorical column.
    """
    rows = []
    for group_val in sorted(df[group_col].dropna().unique()):
        mask = df[group_col] == group_val
        sub = df[mask]
        if len(sub) < 20:
            continue
        stats = calibration_stats(
            sub[y_true_col].values,
            sub[y_prob_col].values,
        )
        rows.append({
            "group": str(group_val),
            "attribute": group_col,
            "n": len(sub),
            "ece": stats["ece"],
            "brier": stats["brier"],
        })
    return pd.DataFrame(rows)


# ── Perturbation testing ──────────────────────────────────────────────────────

def perturbation_test(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    features_to_perturb: list,
    noise_levels: tuple = (0.10, 0.20),
    fairness_df: pd.DataFrame = None,
    protected_col: str = "derived_race",
    reference_group: str = "White",
) -> pd.DataFrame:
    """
    Inject multiplicative noise into continuous features and measure
    AUC delta vs. unperturbed baseline.

    Parameters
    ----------
    model              : fitted sklearn-compatible model (predict_proba)
    X_test             : test feature matrix (DataFrame)
    y_test             : true labels
    features_to_perturb: list of feature names to perturb
    noise_levels       : fractional noise magnitudes to test
    fairness_df        : optional DataFrame with protected attributes for AIR delta
    protected_col      : protected attribute column in fairness_df
    reference_group    : reference group for AIR computation

    Returns
    -------
    DataFrame of perturbation results
    """
    np.random.seed(42)
    baseline_prob = model.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_prob)

    rows = []
    for feat in features_to_perturb:
        if feat not in X_test.columns:
            continue
        for level in noise_levels:
            X_pert = X_test.copy()
            noise = np.random.uniform(1 - level, 1 + level, size=len(X_test))
            X_pert[feat] = X_pert[feat] * noise

            pert_prob = model.predict_proba(X_pert)[:, 1]
            pert_auc = roc_auc_score(y_test, pert_prob)
            auc_delta = round(pert_auc - baseline_auc, 4)

            row = {
                "feature": feat,
                "noise_level": f"±{int(level * 100)}%",
                "baseline_auc": round(baseline_auc, 4),
                "perturbed_auc": round(pert_auc, 4),
                "auc_delta": auc_delta,
                "auc_stable": abs(auc_delta) < 0.02,
            }

            # Compute AIR delta if fairness_df provided
            if fairness_df is not None and protected_col in fairness_df.columns:
                pert_label = (pert_prob >= 0.5).astype(int)
                ref_mask = fairness_df[protected_col] == reference_group
                ref_rate = pert_label[ref_mask].mean() if ref_mask.sum() > 0 else np.nan

                for group_val in fairness_df[protected_col].unique():
                    if group_val == reference_group:
                        continue
                    g_mask = fairness_df[protected_col] == group_val
                    g_rate = pert_label[g_mask].mean()
                    air = g_rate / ref_rate if ref_rate > 0 else np.nan
                    row[f"air_{str(group_val)[:20]}"] = round(air, 4) if not np.isnan(air) else np.nan

            rows.append(row)

    result = pd.DataFrame(rows)
    return result
