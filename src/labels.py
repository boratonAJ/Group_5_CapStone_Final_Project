"""
labels.py — Single source of truth for HMDA capstone label construction.

Label rule (from capstone spec):
    action_taken in {1, 2}  →  y = 1  (originated / approved-not-accepted)
    action_taken == 3       →  y = 0  (denied)
    all other values        →  dropped (withdrawn, incomplete, purchased, etc.)

This module is the ONLY place the label logic lives.
Every notebook imports from here to guarantee consistency.
"""

import pandas as pd
import numpy as np

LABEL_MAP = {1: 1, 2: 1, 3: 0}

ACTION_TAKEN_MEANINGS = {
    1: "Loan originated → y=1",
    2: "Application approved, not accepted → y=1",
    3: "Application denied → y=0",
    4: "Application withdrawn by applicant → DROPPED",
    5: "File closed for incompleteness → DROPPED",
    6: "Purchased loan → DROPPED",
    7: "Preapproval request denied → DROPPED",
    8: "Preapproval request approved but not accepted → DROPPED",
}


def apply_label(df: pd.DataFrame, col: str = "action_taken") -> pd.DataFrame:
    """
    Apply HMDA capstone label rule.

    Parameters
    ----------
    df  : DataFrame containing the action_taken column
    col : name of the action_taken column (default 'action_taken')

    Returns
    -------
    DataFrame with a new 'y' column (int); rows with unmappable action_taken dropped.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    n_raw = len(df)
    df = df.copy()

    df[col] = pd.to_numeric(df[col], errors="coerce")
    df["y"] = df[col].map(LABEL_MAP)

    n_mappable = df["y"].notna().sum()
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    df["y"] = df["y"].astype(int)
    n_final = len(df)
    n_positive = df["y"].sum()

    print("=" * 55)
    print("LABEL CONSTRUCTION LOG")
    print("=" * 55)
    print(f"  Raw rows:           {n_raw:>12,}")
    print(f"  Mappable rows:      {n_mappable:>12,}  ({n_mappable/n_raw:.1%})")
    print(f"  Dropped rows:       {n_raw - n_final:>12,}  (action_taken not in {{1,2,3}})")
    print(f"  Final rows:         {n_final:>12,}")
    print(f"  Positive (y=1):     {n_positive:>12,}  ({n_positive/n_final:.3f})")
    print(f"  Negative (y=0):     {n_final - n_positive:>12,}  ({1 - n_positive/n_final:.3f})")
    print("=" * 55)
    return df


def label_distribution_table(df: pd.DataFrame, col: str = "action_taken") -> pd.DataFrame:
    """
    Return value counts for action_taken with label assignment and retention flag.
    Useful for auditing the label construction step.
    """
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    counts = df[col].value_counts().sort_index()
    result = counts.reset_index()
    result.columns = [col, "count"]
    result["meaning"] = result[col].map(ACTION_TAKEN_MEANINGS).fillna("Unknown")
    result["label_assigned"] = result[col].map(LABEL_MAP)
    result["retained"] = result[col].isin(LABEL_MAP.keys())
    result["pct_of_total"] = (result["count"] / result["count"].sum()).round(4)
    return result
