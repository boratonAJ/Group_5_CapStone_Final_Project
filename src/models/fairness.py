from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix


def _group_error_rates(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    return fpr, fnr


def _two_prop_z_test(p1: float, n1: int, p2: float, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return float("nan")
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan")
    z = (p1 - p2) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def fairness_by_group(
    y_true: pd.Series,
    y_pred: pd.Series,
    group_series: pd.Series,
    min_group_size: int = 100,
) -> pd.DataFrame:
    """Compute group-level fairness metrics with one reference group."""

    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": group_series.fillna("Unknown")})
    group_stats = (
        frame.groupby("group", dropna=False)
        .agg(n=("y_true", "size"), selection_rate=("y_pred", "mean"), base_rate=("y_true", "mean"))
        .reset_index()
    )

    group_stats = group_stats[group_stats["n"] >= min_group_size].copy()
    if group_stats.empty:
        return pd.DataFrame()

    ref_idx = group_stats["selection_rate"].idxmax()
    ref_group = group_stats.loc[ref_idx, "group"]
    ref_sr = float(group_stats.loc[ref_idx, "selection_rate"])
    ref_n = int(group_stats.loc[ref_idx, "n"])

    out_rows = []
    for _, row in group_stats.iterrows():
        group = row["group"]
        g = frame[frame["group"] == group]
        g_fpr, g_fnr = _group_error_rates(g["y_true"], g["y_pred"])

        sr = float(row["selection_rate"])
        n = int(row["n"])
        p_val = _two_prop_z_test(sr, n, ref_sr, ref_n)

        pooled_sd = np.sqrt((sr * (1 - sr) + ref_sr * (1 - ref_sr)) / 2)
        smd = (sr - ref_sr) / pooled_sd if pooled_sd else 0.0

        out_rows.append(
            {
                "group": group,
                "n": n,
                "selection_rate": sr,
                "base_rate": float(row["base_rate"]),
                "reference_group": ref_group,
                "air": (sr / ref_sr) if ref_sr else np.nan,
                "me": sr - ref_sr,
                "smd": float(smd),
                "p_value_selection_rate_vs_ref": p_val,
                "fpr": g_fpr,
                "fnr": g_fnr,
            }
        )

    return pd.DataFrame(out_rows).sort_values("selection_rate", ascending=False).reset_index(drop=True)


def intersectional_fairness(
    y_true: pd.Series,
    y_pred: pd.Series,
    protected: pd.DataFrame,
    cols: list[str],
    min_group_size: int = 100,
) -> pd.DataFrame:
    """Compute fairness metrics for an intersectional attribute key."""

    if not cols:
        return pd.DataFrame()

    key = protected[cols].astype(str).agg("|".join, axis=1)
    return fairness_by_group(y_true=y_true, y_pred=y_pred, group_series=key, min_group_size=min_group_size)
