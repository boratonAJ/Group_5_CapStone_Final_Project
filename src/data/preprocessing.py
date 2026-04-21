from __future__ import annotations

import re

import pandas as pd


def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the DataFrame with normalized column names."""

    cleaned = dataframe.copy()
    cleaned.columns = [
        re.sub(r"[^0-9a-zA-Z_]+", "_", str(column).strip().lower()).strip("_")
        for column in cleaned.columns
    ]
    return cleaned


def basic_clean(dataframe: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Apply lightweight, reusable cleaning steps to a DataFrame."""

    cleaned = standardize_column_names(dataframe)
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned
