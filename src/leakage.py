"""
leakage.py — Post-decision feature deny-list for HMDA capstone.

These features are ONLY populated after the lending decision has been made
(i.e., they are downstream of action_taken). Including them in training would
cause target leakage: the model would achieve artificially high AUC by learning
from information it would not have at prediction time.

Rationale per feature is documented inline.
"""

import pandas as pd

POST_DECISION_FEATURES = [
    # Rate and cost features — populated only for originated loans
    "interest_rate",          # Set at origination; NA for denied applications
    "rate_spread",            # HMDA-required only for originated loans above threshold
    "total_loan_costs",       # Post-origination cost disclosure
    "total_points_and_fees",  # Post-origination
    "origination_charges",    # Post-origination
    "discount_points",        # Post-origination
    "lender_credits",         # Post-origination
    "loan_costs",             # Post-origination (if present)
    # Loan structure features — populated only for originated loans
    "prepayment_penalty_term",  # Post-origination; NA for denied
    "intro_rate_period",        # Post-origination; NA for denied
    # Denial reason fields — populated ONLY for denied applications
    "denial_reason_1",  # Directly encodes the outcome; trivially leaks y=0
    "denial_reason_2",
    "denial_reason_3",
    "denial_reason_4",
]

PROXY_RISK_FEATURES = {
    "census_tract": {
        "risk_level": "High",
        "justification": (
            "Census tract encodes residential segregation patterns. "
            "Highly correlated with race/ethnicity due to historical redlining. "
            "Treat as High proxy risk; monitor SHAP rank and correlation."
        ),
    },
    "tract_minority_population_percent": {
        "risk_level": "High",
        "justification": (
            "Direct demographic encoding of tract-level minority share. "
            "Near-perfect proxy for race if used as a continuous feature."
        ),
    },
    "derived_msa_md": {
        "risk_level": "Medium",
        "justification": (
            "MSA/MD encodes regional income and demographic composition. "
            "Correlated with race/ethnicity due to geographic sorting."
        ),
    },
    "ffiec_msa_md_median_family_income": {
        "risk_level": "Medium",
        "justification": (
            "Tract-level income reflects structural income inequality. "
            "Correlated with race; use with caution."
        ),
    },
    "tract_to_msa_income_percentage": {
        "risk_level": "Medium",
        "justification": (
            "Relative income of tract vs. MSA; encodes neighborhood affluence "
            "which correlates with racial composition."
        ),
    },
    "income": {
        "risk_level": "Medium",
        "justification": (
            "Applicant income is a legitimate creditworthiness indicator but "
            "correlates with race due to structural inequity. Retained; monitored."
        ),
    },
    "applicant_age": {
        "risk_level": "Low",
        "justification": (
            "Age is a protected attribute under ECOA. Monitor model reliance."
        ),
    },
    "state_code": {
        "risk_level": "Medium",
        "justification": (
            "State encodes regional demographic composition. "
            "May proxy for race in some contexts."
        ),
    },
    "county_code": {
        "risk_level": "Medium",
        "justification": "County-level demographic encoding similar to MSA."
    },
}

IDENTIFIER_FEATURES = [
    "lei",            # Lender identifier — not a creditworthiness predictor
    "activity_year",  # Constant in this dataset (2024)
]


def remove_leakage(df: pd.DataFrame, extra: list = None) -> pd.DataFrame:
    """
    Remove post-decision features from a DataFrame.
    Logs which columns were removed and which were already absent.

    Parameters
    ----------
    df    : input DataFrame
    extra : optional list of additional features to remove

    Returns
    -------
    DataFrame with post-decision features removed.
    """
    to_remove = POST_DECISION_FEATURES + IDENTIFIER_FEATURES
    if extra:
        to_remove = to_remove + extra

    present = [f for f in to_remove if f in df.columns]
    absent = [f for f in to_remove if f not in df.columns]

    print("=" * 55)
    print("LEAKAGE REMOVAL LOG")
    print("=" * 55)
    print(f"  Features removed ({len(present)}):")
    for f in present:
        print(f"    - {f}")
    if absent:
        print(f"  Not in dataset (OK, {len(absent)}): {absent}")
    print(f"  Columns before: {len(df.columns)}")
    df = df.drop(columns=present, errors="ignore")
    print(f"  Columns after:  {len(df.columns)}")
    print("=" * 55)
    return df


def get_proxy_risk_table() -> pd.DataFrame:
    """Return the proxy-risk feature table as a DataFrame."""
    rows = []
    for feat, info in PROXY_RISK_FEATURES.items():
        rows.append({
            "feature": feat,
            "risk_level": info["risk_level"],
            "justification": info["justification"],
        })
    return pd.DataFrame(rows)
