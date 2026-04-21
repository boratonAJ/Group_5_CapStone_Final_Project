from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.engineering import build_binary_target, prepare_modeling_frame
from src.models.fairness import fairness_by_group, intersectional_fairness
from src.models.robustness import drift_report, perturb_numeric_features
from src.models.train import build_models, make_splits, train_model
from src.models.transparency import logistic_coefficients, permutation_importance_table


def _flatten_metrics(model_name: str, split_name: str, metric_dict: dict) -> dict:
    row = {"model": model_name, "split": split_name}
    row.update(metric_dict)
    return row


def _ensure_dirs(project_root: Path) -> dict[str, Path]:
    paths = {
        "data_processed": project_root / "data" / "processed",
        "models": project_root / "models",
        "tables": project_root / "reports" / "tables",
        "figures": project_root / "reports" / "figures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _worst_air_penalty(fairness_tables: dict[str, pd.DataFrame]) -> float:
    mins = []
    for table in fairness_tables.values():
        if not table.empty and "air" in table.columns:
            mins.append(table["air"].min())
    if not mins:
        return 0.0
    worst_air = min(mins)
    return max(0.0, 0.8 - float(worst_air))


def _select_final_model(results: dict[str, dict]) -> str:
    scores = {}
    for name, res in results.items():
        test_auc = res["metrics"]["test"].get("auc", 0.0)
        gap = abs(res["metrics"]["train"].get("auc", 0.0) - test_auc)
        fair_penalty = _worst_air_penalty(res.get("fairness", {}))
        score = float(test_auc) - 0.5 * float(gap) - 0.5 * float(fair_penalty)
        scores[name] = score
    return max(scores, key=scores.get)


def _write_governance_memo(
    output_path: Path,
    selected_model: str,
    model_result: dict,
) -> None:
    worst_air = []
    for key, table in model_result.get("fairness", {}).items():
        if not table.empty and "air" in table.columns:
            worst_air.append((key, float(table["air"].min())))

    robustness = model_result.get("robustness", {})

    lines = [
        "# Governance and Deployment Recommendation",
        "",
        "## Intended Use",
        "This model is a decision-support prototype for analytics and compliance review, not an autonomous underwriting system.",
        "",
        "## Selected Model",
        f"- Model: {selected_model}",
        f"- Validation threshold: {model_result.get('threshold', 0.5):.3f}",
        f"- Test AUC: {model_result['metrics']['test'].get('auc', float('nan')):.4f}",
        "",
        "## Fairness Findings",
    ]

    if worst_air:
        for group_name, air in worst_air:
            lines.append(f"- Worst AIR ({group_name}): {air:.4f}")
        lines.append("- Any AIR below 0.80 should trigger remediation before deployment.")
    else:
        lines.append("- No fairness tables available with minimum group size threshold.")

    lines.extend(
        [
            "",
            "## Robustness and Drift Readiness",
            f"- AUC drop under perturbation: {robustness.get('perturb_auc_drop', float('nan')):.4f}",
            f"- Mean PSI (train vs test numeric): {robustness.get('mean_psi', float('nan')):.4f}",
            f"- Max PSI (train vs test numeric): {robustness.get('max_psi', float('nan')):.4f}",
            "",
            "## Security and Abuse Controls",
            "- Protect raw and processed datasets with access control and provenance logging.",
            "- Restrict model artifact access and track download/use events.",
            "- Add rate limiting and anomaly logging to any scoring interface.",
            "",
            "## Monitoring Plan",
            "- Track AUC, selection rate, FPR, and FNR by key protected groups monthly.",
            "- Trigger review if AIR < 0.80, PSI > 0.20, or AUC degrades by more than 0.03.",
            "- Keep versioned logs of model, threshold, and data snapshots.",
            "",
            "## Limitation",
            "- This analysis reflects historical HMDA outcomes and may encode historical policy effects.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    project_root: Path,
    input_file: Path,
    max_rows: int | None = 300000,
    use_full_dataset: bool = False,
) -> None:
    paths = _ensure_dirs(project_root)

    if not input_file.exists():
        raise FileNotFoundError(f"Input cleaned dataset not found: {input_file}")

    # Load data and build binary target
    df_full = pd.read_csv(input_file, low_memory=False)
    df_full = build_binary_target(df_full, target_col="action_taken")
    
    # Self-contained sampling logic (matches notebook Cell 9 for consistency)
    effective_max_rows = None if use_full_dataset else max_rows
    df = df_full.copy()
    
    if effective_max_rows is not None and len(df_full) > effective_max_rows:
        frac = effective_max_rows / len(df_full)
        df = (
            df_full.groupby("action_taken", group_keys=False)
            .sample(frac=frac, random_state=42)
            .reset_index(drop=True)
        )
        mode = "stratified_sample"
        detail = f"effective_max_rows={effective_max_rows}"
    elif effective_max_rows is None:
        mode = "full_dataset"
        detail = "sampling disabled"
    else:
        mode = "full_dataset_no_sampling_needed"
        detail = f"len(df_full)={len(df_full)} <= effective_max_rows={effective_max_rows}"
    
    print(f"Training mode: {mode}")
    print(f"Mode detail: {detail}")
    print("Working shape:", df.shape)
    print("Working target distribution:")
    print(df["action_taken"].value_counts(normalize=True).sort_index())

    prepared = prepare_modeling_frame(
        df,
        target_col="action_taken",
        drop_columns=["lei", "activity_year", "submission_of_application"],
        max_categorical_cardinality=200,
    )

    split = make_splits(prepared.features, prepared.target, random_state=42)

    protected_val = prepared.protected.loc[split.x_val.index] if not prepared.protected.empty else pd.DataFrame(index=split.x_val.index)
    protected_test = prepared.protected.loc[split.x_test.index] if not prepared.protected.empty else pd.DataFrame(index=split.x_test.index)

    model_defs = build_models(random_state=42)
    results: dict[str, dict] = {}

    metric_rows = []
    for model_name, estimator in model_defs.items():
        result = train_model(model_name=model_name, estimator=estimator, split=split)

        fairness_tables = {}
        for col in protected_test.columns:
            fairness_tables[col] = fairness_by_group(
                y_true=split.y_test,
                y_pred=pd.Series(result["pred"]["test"]["y_pred"], index=split.y_test.index),
                group_series=protected_test[col],
                min_group_size=100,
            )

        ix_cols = [c for c in ["applicant_race_1", "applicant_sex"] if c in protected_test.columns]
        fairness_tables["intersection_race_x_sex"] = intersectional_fairness(
            y_true=split.y_test,
            y_pred=pd.Series(result["pred"]["test"]["y_pred"], index=split.y_test.index),
            protected=protected_test,
            cols=ix_cols,
            min_group_size=100,
        )

        drift = drift_report(split.x_train, split.x_test)
        x_test_perturbed = perturb_numeric_features(split.x_test, noise_scale=0.1, random_state=42)
        y_prob_perturbed = result["pipeline"].predict_proba(x_test_perturbed)[:, 1]
        y_pred_perturbed = (y_prob_perturbed >= result["threshold"]).astype(int)

        base_auc = result["metrics"]["test"]["auc"]
        from src.models.evaluate import classification_metrics  # local import to avoid circular dependency during static checks

        perturbed_metrics = classification_metrics(split.y_test, y_pred_perturbed, y_prob_perturbed)

        result["fairness"] = fairness_tables
        result["robustness"] = {
            "mean_psi": float(drift["psi"].mean()) if not drift.empty else float("nan"),
            "max_psi": float(drift["psi"].max()) if not drift.empty else float("nan"),
            "perturb_auc": float(perturbed_metrics["auc"]),
            "perturb_auc_drop": float(base_auc - perturbed_metrics["auc"]),
        }
        result["drift_table"] = drift

        for split_name in ["train", "val", "test"]:
            metric_rows.append(_flatten_metrics(model_name, split_name, result["metrics"][split_name]))

        results[model_name] = result

    final_model_name = _select_final_model(results)
    final = results[final_model_name]

    perf_df = pd.DataFrame(metric_rows)
    perf_df.to_csv(paths["tables"] / "model_performance.csv", index=False)

    for model_name, result in results.items():
        for key, table in result["fairness"].items():
            if not table.empty:
                table.to_csv(paths["tables"] / f"fairness_{model_name}_{key}.csv", index=False)
        if not result["drift_table"].empty:
            result["drift_table"].to_csv(paths["tables"] / f"drift_{model_name}.csv", index=False)

    summary_rows = []
    for name, result in results.items():
        summary_rows.append(
            {
                "model": name,
                "threshold": result["threshold"],
                "test_auc": result["metrics"]["test"]["auc"],
                "test_accuracy": result["metrics"]["test"]["accuracy"],
                "test_log_loss": result["metrics"]["test"]["log_loss"],
                "worst_air_penalty": _worst_air_penalty(result["fairness"]),
                "mean_psi": result["robustness"]["mean_psi"],
                "perturb_auc_drop": result["robustness"]["perturb_auc_drop"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(paths["tables"] / "model_selection_summary.csv", index=False)

    logit_coeff = logistic_coefficients(final["pipeline"], split.x_test)
    if not logit_coeff.empty:
        logit_coeff.to_csv(paths["tables"] / f"logit_coefficients_{final_model_name}.csv", index=False)

    pi_table = permutation_importance_table(final["pipeline"], split.x_test, split.y_test, random_state=42, top_n=40)
    if not pi_table.empty:
        pi_table.to_csv(paths["tables"] / f"permutation_importance_{final_model_name}.csv", index=False)

    roc_display = RocCurveDisplay.from_predictions(
        split.y_test,
        final["pred"]["test"]["y_prob"],
        name=f"{final_model_name} (test)",
    )
    roc_display.figure_.savefig(paths["figures"] / "roc_curve_final_model.png", dpi=180, bbox_inches="tight")
    plt.close(roc_display.figure_)

    if not pi_table.empty:
        top_plot = pi_table.head(15).iloc[::-1]
        plt.figure(figsize=(10, 6))
        plt.barh(top_plot["feature"], top_plot["importance_mean"])
        plt.xlabel("Permutation Importance (AUC drop)")
        plt.title(f"Top Features: {final_model_name}")
        plt.tight_layout()
        plt.savefig(paths["figures"] / "top_feature_importance_final_model.png", dpi=180)
        plt.close()

    joblib.dump(final["pipeline"], paths["models"] / "final_model.joblib")

    metadata = {
        "selected_model": final_model_name,
        "threshold": final["threshold"],
        "input_file": str(input_file),
        "metrics_test": final["metrics"]["test"],
        "robustness": final["robustness"],
    }
    (paths["models"] / "final_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _write_governance_memo(paths["tables"] / "governance_recommendation.md", final_model_name, final)

    print("Pipeline complete.")
    print(f"Selected model: {final_model_name}")
    print(f"Artifacts written to: {paths['tables'].parent}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Responsible AI HMDA capstone pipeline")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root path",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Path to cleaned HMDA CSV. Defaults to data/processed/hmda_lar_2024_cleaned.csv",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=300000,
        help="Maximum number of rows to use for model training and auditing.",
    )
    parser.add_argument(
        "--use-full-dataset",
        action="store_true",
        help="Use the full cleaned dataset for training (disables sampling).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = args.project_root.resolve()
    default_input = root / "data" / "processed" / "hmda_lar_2024_cleaned.csv"
    input_file = args.input_file.resolve() if args.input_file else default_input
    run(
        project_root=root,
        input_file=input_file,
        max_rows=args.max_rows,
        use_full_dataset=args.use_full_dataset,
    )
