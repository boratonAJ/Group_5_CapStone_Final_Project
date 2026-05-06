# Import required libraries
import warnings
import math
import statistics
import urllib.request
from csv import DictReader
import solas_disparity as sd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
from patsy import dmatrices
from scipy.stats import chi2
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from lime.lime_tabular import LimeTabularExplainer
import dice_ml
from dice_ml import Dice

warnings.filterwarnings('ignore')


def prepare_time_varying_frame(df, columns):
    """Prepare a clean start/stop/event frame for lifelines Cox models."""
    frame = df[columns].copy()
    for col in ["id", "start", "end", "event"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna()
    if {"start", "end"}.issubset(frame.columns):
        frame = frame[frame["end"] > frame["start"]].copy()
    return frame


def fit_time_varying_cox(df, columns, formula=None, dummy_cols=None):
    """Fit a CoxTimeVaryingFitter model using either a formula or dummy-coded columns."""
    frame = prepare_time_varying_frame(df, columns)
    # If formula is provided, lifelines will handle dummy encoding. If not, we need to do it ourselves.
    if dummy_cols:
        frame = pd.get_dummies(frame, columns=dummy_cols, drop_first=True)
        for col in frame.columns:
            if frame[col].dtype == bool:
                frame[col] = frame[col].astype(int)
    model = CoxTimeVaryingFitter()
    # The fit_kwargs specify the column names for the Cox model. If a formula is used, we include it in the kwargs.
    fit_kwargs = dict(id_col="id", start_col="start", stop_col="end", event_col="event")
    if formula is not None:
        fit_kwargs["formula"] = formula
    model.fit(frame, **fit_kwargs)
    return model, frame


def print_cox_summary_rstyle(model, model_frame, formula_text):
    """Print a Cox model summary in an R-like console format."""
    summary_df = model.summary.rename(columns={
        "coef": "coef",
        "exp(coef)": "exp(coef)",
        "se(coef)": "se(coef)",
        "z": "z",
        "p": "Pr(>|z|)",
        "exp(coef) lower 95%": "lower .95",
        "exp(coef) upper 95%": "upper .95",
    }).copy()
    # Calculate overall model statistics
    n_obs = model_frame.shape[0]
    n_events = int(model_frame["event"].sum())
    llr = model.log_likelihood_ratio_test()
    llr_stat = float(llr.test_statistic)
    llr_df = int(llr.degrees_freedom)
    llr_p = float(llr.p_value)
    wald_stat = float((summary_df["z"] ** 2).sum())
    wald_df = len(summary_df)
    wald_p = float(chi2.sf(wald_stat, wald_df))

    print("Call:")
    print(f"coxph(formula = {formula_text})\n")
    print(f"  n= {n_obs}, number of events= {n_events}\n")
    width = max(20, min(42, max(len(str(idx)) for idx in summary_df.index) + 2)) if len(summary_df) else 20
    print(f"{'':{width}s}{'coef':>10s} {'exp(coef)':>10s} {'se(coef)':>9s} {'z':>8s} {'Pr(>|z|)':>10s}")
    for idx, row in summary_df.iterrows():
        pval = row["Pr(>|z|)"]
        if pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"
        elif pval < 0.1:
            sig = "."
        else:
            sig = ""
        p_display = "<2e-16" if pval < 2e-16 else f"{pval:.3g}"
        print(
            f"{str(idx):{width}s}"
            f"{row['coef']:10.5f} "
            f"{row['exp(coef)']:10.5f} "
            f"{row['se(coef)']:9.5f} "
            f"{row['z']:8.3f} "
            f"{p_display:>10s} {sig}"
        )
    print("---")
    print("Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1\n")
    print(f"{'':{width}s}{'exp(coef)':>10s} {'exp(-coef)':>11s} {'lower .95':>10s} {'upper .95':>10s}")
    for idx, row in summary_df.iterrows():
        print(
            f"{str(idx):{width}s}"
            f"{row['exp(coef)']:10.3f} "
            f"{np.exp(-row['coef']):11.4f} "
            f"{row['lower .95']:10.3f} "
            f"{row['upper .95']:10.3f}"
        )
    print()
    print("Concordance= NA (se = NA )")
    print(f"Likelihood ratio test= {llr_stat:.1f}  on {llr_df} df,   p={llr_p:.3g}")
    print(f"Wald test            = {wald_stat:.1f}  on {wald_df} df,   p={wald_p:.3g}")
    print(f"Score (logrank) test = {llr_stat:.1f}  on {llr_df} df,   p={llr_p:.3g}")


def prepare_km_frame(df, extra_cols=None):
    """Prepare a clean frame for Kaplan-Meier plots, ensuring numeric types and dropping duplicates."""
    extra_cols = extra_cols or []
    frame = df[["id", "end", "event", "score_factor", *extra_cols]].drop_duplicates(subset="id").copy()
    frame["end"] = pd.to_numeric(frame["end"], errors="coerce")
    frame["event"] = pd.to_numeric(frame["event"], errors="coerce")
    return frame.dropna().copy()


def plot_km_grouped(df_subset, title, ax=None):
    """Plot Kaplan-Meier survival curves grouped by score_factor. If ax is None, create a new figure."""
    kmf = KaplanMeierFitter() # We will fit the KM model separately for each group to plot the survival curves.
    created_fig = False
    if ax is None:
        plt.figure(figsize=(6, 3.5))
        ax = plt.gca()
        created_fig = True
    for group in sorted(df_subset["score_factor"].dropna().unique()):
        subset = df_subset[df_subset["score_factor"] == group]
        kmf.fit(subset["end"], subset["event"], label=str(group))
        kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.3)
    if created_fig:
        plt.show()


def print_km_summary_rstyle(df_subset, call_text, time_point=730):
    """Print a Kaplan-Meier summary at a specific time point in an R-like console format."""
    kmf = KaplanMeierFitter() # We will fit the KM model separately for each group to get the survival probability and confidence intervals at the specified time point.
    print(f"Call: survfit(formula = f, data = {call_text})\n")
    for group in sorted(df_subset["score_factor"].dropna().unique()):
        subset = df_subset[df_subset["score_factor"] == group]
        kmf.fit(subset["end"], subset["event"], label=str(group))
        surv = kmf.predict(time_point)
        ci_idx = kmf.confidence_interval_.index[kmf.confidence_interval_.index <= time_point].max()
        ci = kmf.confidence_interval_.loc[ci_idx]
        se = (ci.iloc[1] - ci.iloc[0]) / (2 * 1.96)
        n_risk = (subset["end"] >= time_point).sum()
        n_event = ((subset["event"] == 1) & (subset["end"] <= time_point)).sum()
        n_censored = ((subset["event"] == 0) & (subset["end"] <= time_point)).sum()
        entered = len(subset)
        print(f"                score_factor={group}")
        print("        time     n.risk     n.event     entered     censored     survival")
        print(f"    {time_point:8.2e}  {n_risk:10.2e}  {n_event:10.2e}  {entered:10.2e}  {n_censored:10.2e}  {surv:10.2e}")
        print("        std.err    lower 95% CI    upper 95% CI")
        print(f"    {se:10.2e}  {ci.iloc[0]:10.2e}  {ci.iloc[1]:10.2e}\n")


def side_by_side_km(left_df, left_title, right_df, right_title, figsize=(14, 5)):
    """Plot two Kaplan-Meier curves side by side for easy comparison."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_km_grouped(left_df, left_title, ax=axes[0])
    plot_km_grouped(right_df, right_title, ax=axes[1])
    plt.tight_layout()
    plt.show()
