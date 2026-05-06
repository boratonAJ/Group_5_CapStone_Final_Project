"""
monitoring.py — Review-trigger evaluation logic for HMDA capstone.

Defines the monitoring playbook thresholds and evaluates whether current
metrics are in the Green, Yellow, or Red zone.
"""

import numpy as np
import pandas as pd
from datetime import datetime


# ── Review-trigger thresholds ─────────────────────────────────────────────────
# These are the operational thresholds documented in the deployment recommendation.
# RED triggers require immediate action; YELLOW triggers require increased monitoring.

REVIEW_TRIGGERS = [
    {
        "metric": "AUC (overall)",
        "frequency": "Monthly",
        "green_threshold": ">= 0.72",
        "yellow_threshold": "0.68 – 0.72",
        "red_threshold": "< 0.68",
        "green_fn": lambda v: v >= 0.72,
        "yellow_fn": lambda v: 0.68 <= v < 0.72,
        "red_fn": lambda v: v < 0.68,
        "owner": "ML Engineer",
        "action_yellow": "Review feature distributions; check for drift",
        "action_red": "Pause deployment; initiate full retrain review",
    },
    {
        "metric": "AIR — Black applicants",
        "frequency": "Monthly",
        "green_threshold": ">= 0.80",
        "yellow_threshold": "0.70 – 0.80",
        "red_threshold": "< 0.70",
        "green_fn": lambda v: v >= 0.80,
        "yellow_fn": lambda v: 0.70 <= v < 0.80,
        "red_fn": lambda v: v < 0.70,
        "owner": "Fairness Officer",
        "action_yellow": "Investigate source of disparity; report to compliance",
        "action_red": "IMMEDIATE PAUSE; fairness re-audit required",
    },
    {
        "metric": "AIR — Hispanic or Latino applicants",
        "frequency": "Monthly",
        "green_threshold": ">= 0.80",
        "yellow_threshold": "0.70 – 0.80",
        "red_threshold": "< 0.70",
        "green_fn": lambda v: v >= 0.80,
        "yellow_fn": lambda v: 0.70 <= v < 0.80,
        "red_fn": lambda v: v < 0.70,
        "owner": "Fairness Officer",
        "action_yellow": "Investigate source of disparity; report to compliance",
        "action_red": "IMMEDIATE PAUSE; fairness re-audit required",
    },
    {
        "metric": "AIR — Female applicants",
        "frequency": "Monthly",
        "green_threshold": ">= 0.80",
        "yellow_threshold": "0.70 – 0.80",
        "red_threshold": "< 0.70",
        "green_fn": lambda v: v >= 0.80,
        "yellow_fn": lambda v: 0.70 <= v < 0.80,
        "red_fn": lambda v: v < 0.70,
        "owner": "Fairness Officer",
        "action_yellow": "Investigate source of disparity; report to compliance",
        "action_red": "IMMEDIATE PAUSE; fairness re-audit required",
    },
    {
        "metric": "AIR — Applicant sex",
        "frequency": "Monthly",
        "green_threshold": ">= 0.80",
        "yellow_threshold": "0.70 – 0.80",
        "red_threshold": "< 0.70",
        "green_fn": lambda v: v >= 0.80,
        "yellow_fn": lambda v: 0.70 <= v < 0.80,
        "red_fn": lambda v: v < 0.70,
        "owner": "Fairness Officer",
        "action_yellow": "Investigate source of disparity; report to compliance",
        "action_red": "IMMEDIATE PAUSE; fairness re-audit required",
    },
    {
        "metric": "AIR — Applicant age",
        "frequency": "Monthly",
        "green_threshold": ">= 0.80",
        "yellow_threshold": "0.70 – 0.80",
        "red_threshold": "< 0.70",
        "green_fn": lambda v: v >= 0.80,
        "yellow_fn": lambda v: 0.70 <= v < 0.80,
        "red_fn": lambda v: v < 0.70,
        "owner": "Fairness Officer",
        "action_yellow": "Investigate source of disparity; report to compliance",
        "action_red": "IMMEDIATE PAUSE; fairness re-audit required",
    },
    {
        "metric": "PSI — loan_amount",
        "frequency": "Quarterly",
        "green_threshold": "< 0.10",
        "yellow_threshold": "0.10 – 0.25",
        "red_threshold": "> 0.25",
        "green_fn": lambda v: v < 0.10,
        "yellow_fn": lambda v: 0.10 <= v <= 0.25,
        "red_fn": lambda v: v > 0.25,
        "owner": "Data Owner",
        "action_yellow": "Review loan amount distribution; assess if product mix changed",
        "action_red": "Feature re-engineering review; model retrain evaluation",
    },
    {
        "metric": "PSI — income",
        "frequency": "Quarterly",
        "green_threshold": "< 0.10",
        "yellow_threshold": "0.10 – 0.25",
        "red_threshold": "> 0.25",
        "green_fn": lambda v: v < 0.10,
        "yellow_fn": lambda v: 0.10 <= v <= 0.25,
        "red_fn": lambda v: v > 0.25,
        "owner": "Data Owner",
        "action_yellow": "Review income distribution; check for reporting changes",
        "action_red": "Feature re-engineering review; model retrain evaluation",
    },
    {
        "metric": "Subgroup ECE gap (max across race groups)",
        "frequency": "Quarterly",
        "green_threshold": "< 0.05",
        "yellow_threshold": "0.05 – 0.10",
        "red_threshold": "> 0.10",
        "green_fn": lambda v: v < 0.05,
        "yellow_fn": lambda v: 0.05 <= v <= 0.10,
        "red_fn": lambda v: v > 0.10,
        "owner": "Fairness Officer",
        "action_yellow": "Investigate calibration gap; consider isotonic recalibration",
        "action_red": "Recalibration required before continued deployment",
    },
    {
        "metric": "Approval rate change vs. baseline (absolute)",
        "frequency": "Monthly",
        "green_threshold": "+/- 5%",
        "yellow_threshold": "+/- 5-10%",
        "red_threshold": "+/- > 10%",
        "green_fn": lambda v: abs(v) < 0.05,
        "yellow_fn": lambda v: 0.05 <= abs(v) <= 0.10,
        "red_fn": lambda v: abs(v) > 0.10,
        "owner": "Compliance",
        "action_yellow": "Report to compliance; assess business driver",
        "action_red": "Compliance review; potential regulatory notification",
    },
]


def evaluate_triggers(metric_values: dict) -> pd.DataFrame:
    """
    Evaluate current metric values against review triggers.

    Parameters
    ----------
    metric_values : dict {metric_name: current_value}

    Returns
    -------
    DataFrame with status (Green/Yellow/Red) for each monitored metric
    """
    rows = []
    for trigger in REVIEW_TRIGGERS:
        metric = trigger["metric"]
        val = metric_values.get(metric, np.nan)
        if np.isnan(val) if isinstance(val, float) else False:
            status = "Unknown"
            action = "Compute metric"
        elif trigger["red_fn"](val):
            status = "🔴 Red"
            action = trigger["action_red"]
        elif trigger["yellow_fn"](val):
            status = "🟡 Yellow"
            action = trigger["action_yellow"]
        else:
            status = "🟢 Green"
            action = "Continue monitoring"

        rows.append({
            "metric": metric,
            "frequency": trigger["frequency"],
            "current_value": val,
            "green_threshold": trigger["green_threshold"],
            "yellow_threshold": trigger["yellow_threshold"],
            "red_threshold": trigger["red_threshold"],
            "status": status,
            "owner": trigger["owner"],
            "required_action": action,
        })
    return pd.DataFrame(rows)


def get_playbook_table() -> pd.DataFrame:
    """
    Return the static monitoring playbook table (without current values).
    Suitable for documentation.
    """
    rows = []
    for trigger in REVIEW_TRIGGERS:
        rows.append({
            "metric": trigger["metric"],
            "frequency": trigger["frequency"],
            "green": trigger["green_threshold"],
            "yellow": trigger["yellow_threshold"],
            "red": trigger["red_threshold"],
            "owner": trigger["owner"],
            "action_yellow": trigger["action_yellow"],
            "action_red": trigger["action_red"],
        })
    return pd.DataFrame(rows)
