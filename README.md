# Group 5 Capstone Project

This folder is a clean data-science project scaffold for the Group 5 capstone.

Starter assets:

- [Notebook starter](notebooks/Group_5_Capstone_Starter.ipynb)
- [README template](README_TEMPLATE.md)
- [Python requirements](requirements.txt)

## Suggested Structure

- `data/raw/` - original, immutable source data
- `data/interim/` - intermediate data prepared during cleaning
- `data/processed/` - analysis-ready datasets
- `notebooks/` - exploratory analysis and reporting notebooks
- `src/` - reusable source code for data, features, models, and visualization
- `models/` - saved model artifacts
- `reports/figures/` - charts and visuals for the final report
- `reports/tables/` - exported tables for the final report
- `docs/` - project notes, scope, and governance documents
- `references/` - papers, links, and source references
- `scripts/` - runnable scripts for repeatable tasks
- `tests/` - unit or validation tests

## Working Conventions

- Keep raw data unchanged.
- Put reusable logic in `src/`, not in notebooks.
- Name notebooks and scripts by stage or purpose.
- Save final figures and tables in `reports/`.

## HMDA LAR Loading Highlight

The starter notebook includes a robust loading flow for the 2024 HMDA LAR dataset.

Expected raw input location:

- `data/raw/hmda_lar_2024.zip` (optional)
- `data/raw/hmda_lar_2024.csv`
- `data/raw/hmda_lar_2024.txt`
- `data/raw/hmda_lar_2024.tsv`
- `data/raw/hmda_lar_2024.psv`

Pipeline behavior:

1. Optionally unzip `hmda_lar_2024.zip` into `data/raw/`.
2. Detect the available file extension from supported tabular formats.
3. Detect delimiter before loading (extension defaults plus `csv.Sniffer` validation).
4. Load using `pandas.read_csv` with the detected separator.
5. Clean and export to `data/processed/hmda_lar_2024_cleaned.csv`.

This prevents incorrect parsing when the raw file is tab-delimited, pipe-delimited, or semicolon-delimited rather than comma-delimited.

## End-to-End Capstone Pipeline

This repository now includes a full PRD-aligned Responsible ML pipeline for the HMDA binary classification task.

Technical documentation:

- [Full technical pipeline documentation](docs/TECHNICAL_PIPELINE_DOCUMENTATION.md)

Entry point:

- `scripts/run_capstone_pipeline.py`

Core capabilities:

- Binary label construction from `action_taken` (`1,2 -> 1`, `3 -> 0`, others dropped)
- Reproducible train/validation/test splits
- Baseline interpretable model: Logistic Regression
- Higher-capacity comparison model: Random Forest
- Predictive evaluation: AUC, accuracy, log loss, F1, confusion-derived rates
- Fairness audit: group-level AIR, ME, SMD, FPR, FNR and significance testing
- Intersectional fairness: race x sex (when present)
- Robustness checks: perturbation sensitivity and drift indicators (PSI, KS)
- Transparency outputs: logistic coefficients and permutation importance
- Governance output: model-selection and deployment recommendation memo

### Run

From the project root:

```bash
python scripts/run_capstone_pipeline.py
```

Or provide an explicit cleaned input file:

```bash
python scripts/run_capstone_pipeline.py --input-file data/processed/hmda_lar_2024_cleaned.csv
```

### Expected Outputs

- Model artifact: `models/final_model.joblib`
- Metadata: `models/final_model_metadata.json`
- Tables: `reports/tables/model_performance.csv`
- Fairness tables: `reports/tables/fairness_<model>_<group>.csv`
- Drift tables: `reports/tables/drift_<model>.csv`
- Model selection summary: `reports/tables/model_selection_summary.csv`
- Governance memo: `reports/tables/governance_recommendation.md`
- Figures: `reports/figures/roc_curve_final_model.png`
- Figures: `reports/figures/top_feature_importance_final_model.png`
