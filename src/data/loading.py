from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(file_path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """Load a CSV file from disk and return it as a DataFrame."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path, **read_csv_kwargs)


def save_csv(dataframe: pd.DataFrame, file_path: str | Path, **to_csv_kwargs) -> None:
    """Save a DataFrame to CSV, creating the parent directory if needed."""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False, **to_csv_kwargs)
