from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def _safe_feature_names(pipeline, x_sample: pd.DataFrame) -> np.ndarray:
    prep = pipeline.named_steps["preprocess"]
    try:
        return prep.get_feature_names_out()
    except Exception:
        return np.array([f"feature_{i}" for i in range(prep.transform(x_sample).shape[1])])


def logistic_coefficients(pipeline, x_sample: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame()

    names = _safe_feature_names(pipeline, x_sample)
    coef = model.coef_.ravel()
    out = pd.DataFrame({"feature": names, "coefficient": coef})
    out["abs_coefficient"] = out["coefficient"].abs()
    return out.sort_values("abs_coefficient", ascending=False).head(top_n).reset_index(drop=True)


def permutation_importance_table(
    pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    top_n: int = 30,
) -> pd.DataFrame:
    names = np.array(x.columns.astype(str).tolist())
    pi = permutation_importance(
        pipeline,
        x,
        y,
        scoring="roc_auc",
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
    )

    out = pd.DataFrame(
        {
            "feature": names,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
        }
    )
    return out.sort_values("importance_mean", ascending=False).head(top_n).reset_index(drop=True)
