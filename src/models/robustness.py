from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(train_series: pd.Series, test_series: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for one numeric feature."""

    train = train_series.dropna().astype(float)
    test = test_series.dropna().astype(float)

    if train.empty or test.empty:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(train, quantiles))
    if len(breakpoints) < 3:
        return 0.0

    train_counts, _ = np.histogram(train, bins=breakpoints)
    test_counts, _ = np.histogram(test, bins=breakpoints)

    train_pct = np.clip(train_counts / train_counts.sum(), 1e-6, None)
    test_pct = np.clip(test_counts / test_counts.sum(), 1e-6, None)
    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))


def drift_report(x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
    """Compute PSI and KS for numeric columns as drift-readiness indicators."""

    numeric_cols = x_train.select_dtypes(include=["number", "bool"]).columns
    rows = []
    for col in numeric_cols:
        tr = x_train[col]
        te = x_test[col] if col in x_test.columns else pd.Series(dtype=float)
        if te.empty:
            continue
        try:
            ks_stat = float(ks_2samp(tr.dropna().astype(float), te.dropna().astype(float)).statistic)
        except Exception:
            ks_stat = float("nan")
        rows.append({"feature": col, "psi": psi(tr, te), "ks_stat": ks_stat})

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def perturb_numeric_features(x: pd.DataFrame, noise_scale: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """Return a copy with Gaussian perturbation on numeric columns."""

    rng = np.random.default_rng(random_state)
    out = x.copy()
    numeric_cols = out.select_dtypes(include=["number", "bool"]).columns

    for col in numeric_cols:
        std = float(out[col].std()) if out[col].notna().any() else 0.0
        if std == 0.0:
            continue
        noise = rng.normal(0, noise_scale * std, size=len(out))
        out[col] = out[col].astype(float) + noise

    return out
