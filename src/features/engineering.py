from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROTECTED_COLUMNS = [
    "applicant_ethnicity_1",
    "applicant_race_1",
    "applicant_sex",
    "co_applicant_ethnicity_1",
    "co_applicant_race_1",
    "co_applicant_sex",
]


@dataclass
class PreparedData:
    features: pd.DataFrame
    target: pd.Series
    protected: pd.DataFrame


def build_binary_target(df: pd.DataFrame, target_col: str = "action_taken") -> pd.DataFrame:
    """Map HMDA action_taken to binary target and filter unsupported outcomes."""

    if target_col not in df.columns:
        raise ValueError(f"Missing required target column: {target_col}")

    out = df.copy()
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")

    unique_values = set(out[target_col].dropna().unique().tolist())
    if unique_values.issubset({0, 1}):
        out[target_col] = out[target_col].astype(int)
        return out

    out = out[out[target_col].isin([1, 2, 3])].copy()
    out[target_col] = out[target_col].map({1: 1, 2: 1, 3: 0}).astype(int)
    return out


def prepare_modeling_frame(
    df: pd.DataFrame,
    target_col: str = "action_taken",
    drop_columns: list[str] | None = None,
    max_categorical_cardinality: int = 200,
) -> PreparedData:
    """Split a DataFrame into model features, target, and protected attributes."""

    drop_columns = drop_columns or []
    frame = df.copy()
    protected_cols = [c for c in PROTECTED_COLUMNS if c in frame.columns]

    y = frame[target_col].astype(int)
    protected = frame[protected_cols].copy() if protected_cols else pd.DataFrame(index=frame.index)

    forbidden = set([target_col, *drop_columns, *protected_cols])
    x = frame[[c for c in frame.columns if c not in forbidden]].copy()

    # Remove non-informative columns before encoding.
    x = x.dropna(axis=1, how="all")
    constant_cols = [c for c in x.columns if x[c].nunique(dropna=True) <= 1]
    if constant_cols:
        x = x.drop(columns=constant_cols)

    # Drop extreme-cardinality categorical columns to keep memory bounded.
    cat_cols = x.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    high_card = [
        c for c in cat_cols if x[c].nunique(dropna=True) > max_categorical_cardinality
    ]
    if high_card:
        x = x.drop(columns=high_card)

    return PreparedData(features=x, target=y, protected=protected)


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Build sklearn preprocessing for mixed-type HMDA tabular features."""

    numeric_cols = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    max_categories=50,
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )
