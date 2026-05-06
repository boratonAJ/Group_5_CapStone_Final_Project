"""
explain.py — SHAP-based explainability and proxy-risk analysis for HMDA capstone.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def compute_shap_values(model, X: pd.DataFrame, max_display: int = 20):
    """
    Compute TreeSHAP values for a tree-based model.
    Returns shap_values array and shap.Explanation object.
    """
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return explainer, shap_values


def shap_summary_bar(shap_values, X: pd.DataFrame, max_display: int = 20,
                     save_path: str = None):
    """
    SHAP bar plot: mean |SHAP value| per feature (top N).
    """
    import shap
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title("Mean |SHAP Value| — Top Feature Importances", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[explain] Saved: {save_path}")
    plt.close()


def shap_beeswarm(shap_values, max_display: int = 20, save_path: str = None):
    """
    SHAP beeswarm plot: distribution of SHAP values per feature.
    """
    import shap
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Beeswarm — Feature Effect Distributions", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[explain] Saved: {save_path}")
    plt.close()


def top_features_table(
    shap_values,
    X: pd.DataFrame,
    proxy_risk_dict: dict = None,
    n: int = 20,
) -> pd.DataFrame:
    """
    Build a ranked table of top N features by mean |SHAP value|.
    Optionally annotate with proxy-risk ratings.

    Parameters
    ----------
    shap_values   : SHAP Explanation object
    X             : feature DataFrame used for SHAP computation
    proxy_risk_dict : dict {feature_name: {'risk_level': ..., 'justification': ...}}
    n             : number of top features to return

    Returns
    -------
    DataFrame: rank, feature, mean_abs_shap, proxy_risk_level, proxy_justification
    """
    shap_vals = shap_values.values
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    mean_abs = np.abs(shap_vals).mean(axis=0)
    feat_names = X.columns.tolist()

    df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).head(n).reset_index(drop=True)
    df["rank"] = df.index + 1

    df["proxy_risk_level"] = "Low"
    df["proxy_justification"] = ""

    if proxy_risk_dict:
        for i, row in df.iterrows():
            feat = row["feature"]
            # Exact match first
            if feat in proxy_risk_dict:
                df.at[i, "proxy_risk_level"] = proxy_risk_dict[feat].get("risk_level", "Low")
                df.at[i, "proxy_justification"] = proxy_risk_dict[feat].get("justification", "")
            else:
                # Prefix match (for one-hot encoded features like loan_type_2)
                for k, v in proxy_risk_dict.items():
                    if feat.startswith(k):
                        df.at[i, "proxy_risk_level"] = v.get("risk_level", "Low")
                        df.at[i, "proxy_justification"] = v.get("justification", "")
                        break

    cols = ["rank", "feature", "mean_abs_shap", "proxy_risk_level", "proxy_justification"]
    return df[[c for c in cols if c in df.columns]]


def proxy_correlation_table(
    X: pd.DataFrame,
    protected_cols: dict,
    top_features: list,
) -> pd.DataFrame:
    """
    Compute Pearson correlation between model features and protected attributes.
    Protected attributes must be label-encoded (numeric) before passing.

    Parameters
    ----------
    X               : feature DataFrame
    protected_cols  : dict {display_name: column_name} — e.g. {'Race': 'race_encoded'}
    top_features    : list of feature names to include

    Returns
    -------
    DataFrame: feature × protected attribute correlations
    """
    rows = []
    for feat in top_features:
        if feat not in X.columns:
            continue
        row = {"feature": feat}
        feat_vals = pd.to_numeric(X[feat], errors="coerce")
        for display_name, col_name in protected_cols.items():
            if col_name not in X.columns:
                continue
            prot_vals = pd.to_numeric(X[col_name], errors="coerce")
            valid = feat_vals.notna() & prot_vals.notna()
            if valid.sum() < 20:
                row[f"corr_{display_name}"] = np.nan
            else:
                corr = feat_vals[valid].corr(prot_vals[valid])
                row[f"corr_{display_name}"] = round(corr, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_counterfactuals(
    model,
    X_denied: pd.DataFrame,
    feature_deltas: dict,
    threshold: float = 0.5,
    n_cases: int = 10,
) -> pd.DataFrame:
    """
    Simple counterfactual generator: for each denied applicant, try
    increasing/decreasing each feature in feature_deltas and check if
    the decision flips.

    Parameters
    ----------
    model         : fitted model with predict_proba
    X_denied      : feature matrix for denied applicants
    feature_deltas: dict {feature_name: delta_value} — changes to test
    threshold     : decision threshold
    n_cases       : number of cases to analyze

    Returns
    -------
    DataFrame with counterfactual flip information per case
    """
    X_sample = X_denied.head(n_cases).copy()
    baseline_probs = model.predict_proba(X_sample)[:, 1]

    rows = []
    for i in range(len(X_sample)):
        for feat, delta in feature_deltas.items():
            if feat not in X_sample.columns:
                continue
            X_cf = X_sample.copy()
            X_cf.iloc[i, X_cf.columns.get_loc(feat)] += delta
            cf_prob = model.predict_proba(X_cf.iloc[[i]])[:, 1][0]
            flip = (baseline_probs[i] < threshold) and (cf_prob >= threshold)
            rows.append({
                "case_index": i,
                "feature_changed": feat,
                "original_value": round(float(X_sample.iloc[i][feat]), 2),
                "counterfactual_value": round(float(X_sample.iloc[i][feat] + delta), 2),
                "delta": delta,
                "original_prob": round(float(baseline_probs[i]), 4),
                "counterfactual_prob": round(float(cf_prob), 4),
                "decision_flipped": flip,
            })

    return pd.DataFrame(rows)
