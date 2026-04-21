# Group 5 Capstone Project

## Project Overview

This repository contains an end-to-end Responsible AI workflow for Home Mortgage Disclosure Act (HMDA) Loan/Application Register (LAR) data. The project is designed to be reproducible, auditable, and governance-ready.

Primary objective:
- Build and evaluate a binary classification pipeline for mortgage outcome analysis using HMDA records.

Responsible AI objective:
- Evaluate not only predictive performance, but also fairness, drift-readiness, robustness under perturbation, and transparency.

## Team Members

- Name 1
- Name 2
- Name 3
- Name 4
- Name 5

## Problem Statement

Mortgage outcome data can reflect both credit risk patterns and historical policy effects. The capstone goal is to build a decision-support model that is technically sound and responsibly evaluated, with explicit checks for fairness and operational risk.

## Scope

In scope:
- Binary target construction from HMDA action outcomes.
- Structured preprocessing for numeric and categorical data.
- Model training and threshold tuning.
- Evaluation on train, validation, and test splits.
- Fairness auditing across protected groups and intersectional slices.
- Drift and perturbation robustness checks.
- Transparency outputs and governance recommendations.

Out of scope:
- Autonomous credit approval decisions.
- Production deployment as a final underwriting system.

## Repository Structure

- data/raw/: source data files
- data/processed/: cleaned and analysis-ready datasets
- notebooks/: exploratory and full pipeline notebooks
- src/: reusable code for data, features, models, and visualization
- models/: exported model artifacts and metadata
- reports/figures/: exported figures
- reports/tables/: exported evaluation, fairness, drift, and governance tables
- docs/: technical documentation and architecture assets
- references/: citations and source references
- scripts/: repeatable execution entry points
- tests/: unit and validation tests

## Quick Start

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Ensure cleaned input data exists at data/processed/hmda_lar_2024_cleaned.csv.
5. Run the pipeline script.

Example commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_capstone_pipeline.py
```

Optional runtime controls:

```bash
python scripts/run_capstone_pipeline.py --max-rows 300000
python scripts/run_capstone_pipeline.py --use-full-dataset
python scripts/run_capstone_pipeline.py --input-file data/processed/hmda_lar_2024_cleaned.csv
```

## Data Sources and Data Management

Primary source:
- HMDA 2024 LAR data.

Expected naming and location:
- data/raw/hmda_lar_2024.zip (optional)
- data/raw/hmda_lar_2024.csv
- data/raw/hmda_lar_2024.txt
- data/raw/hmda_lar_2024.tsv
- data/raw/hmda_lar_2024.psv

Processed output:
- data/processed/hmda_lar_2024_cleaned.csv

Recommended loading flow:
1. If present, unzip hmda_lar_2024.zip into data/raw/.
2. Detect the available raw file among supported extensions.
3. Infer delimiter from extension defaults, then validate with csv.Sniffer.
4. Load with pandas.read_csv using the detected delimiter.
5. Apply cleaning and save the processed file.

Why this matters:
- Prevents silent parsing errors for tab-delimited or pipe-delimited files.
- Keeps preprocessing robust to common tabular text variants.

## Methodology

### Target Construction

The pipeline maps action_taken outcomes as follows:
- 1, 2 -> positive class (1)
- 3 -> negative class (0)
- Other values are excluded

### Modeling Pipeline

- Stratified train/validation/test split (60/20/20).
- Numeric preprocessing: median imputation + standard scaling.
- Categorical preprocessing: most-frequent imputation + one-hot encoding.
- Candidate models:
	- Logistic Regression
	- Random Forest
- Validation threshold tuning using Youden J.

### Evaluation Metrics

- AUC
- Accuracy
- Log loss
- F1
- TPR, TNR, FPR, FNR
- Positive prediction rate

## Responsible AI and Governance

### Fairness Audits

For each protected attribute available in the dataset, the pipeline computes:
- Selection rate and base rate
- AIR (adverse impact ratio)
- Mean effect
- Standardized mean difference
- FPR and FNR by group
- p-value versus reference group (two-proportion z-test)

Intersectional fairness is included for race x sex when available.

### Robustness and Drift

- Drift indicators on numeric features:
	- PSI (Population Stability Index)
	- KS statistic
- Perturbation stress test:
	- Gaussian noise on numeric test features
	- Perturbed AUC and AUC drop tracking

### Transparency

- Logistic coefficient table when the selected model is linear.
- Permutation importance table for feature contribution analysis.

### Governance Output

The pipeline writes a governance memo containing:
- Intended use
- Selection threshold and test performance
- Worst AIR observations
- Drift and robustness summary
- Security and monitoring recommendations

See:
- docs/TECHNICAL_PIPELINE_DOCUMENTATION.md

## Model Selection Logic

The selected model is based on a composite score balancing:
- Test AUC
- Overfit penalty (train-test AUC gap)
- Fairness penalty (AIR below 0.80)

## Outputs

Main generated artifacts:
- models/final_model.joblib
- models/final_model_metadata.json
- reports/tables/model_performance.csv
- reports/tables/model_selection_summary.csv
- reports/tables/fairness_<model>_<group>.csv
- reports/tables/drift_<model>.csv
- reports/tables/governance_recommendation.md
- reports/figures/roc_curve_final_model.png
- reports/figures/top_feature_importance_final_model.png

## Testing and Quality Assurance

Run tests:

```bash
pytest
```

Current test coverage includes:
- Target mapping and preprocessing behavior.
- Fairness table schema validation.

## Project Management and Collaboration

### Working Model

- Use task-based branches for feature work and fixes.
- Keep pull requests focused and reviewable.
- Require clear commit messages and linked issue context.

### Suggested Cadence

- Weekly planning: define goals, deliverables, and owners.
- Mid-week sync: remove blockers and align assumptions.
- End-of-week review: demo outputs, verify metrics, and update risks.

### Suggested Roles

- Data lead: data quality, cleaning assumptions, and provenance.
- Modeling lead: training, evaluation, and threshold strategy.
- Responsible AI lead: fairness, robustness, and governance memo.
- Documentation lead: README, technical docs, and report artifacts.
- QA lead: tests, reproducibility checks, and runbook updates.

### Risk Management

Track and review these risks during execution:
- Data format drift or schema changes.
- Class imbalance effects on calibration and thresholding.
- Fairness degradation for smaller subgroups.
- Performance instability under distribution shift.

## Reproducibility Checklist

- Fixed random seeds for sampling, splits, and model training where supported.
- Deterministic pipeline script entry point in scripts/run_capstone_pipeline.py.
- Versioned artifact outputs in models/ and reports/.
- Explicit dependency list in requirements.txt.

## Documentation Map

- docs/TECHNICAL_PIPELINE_DOCUMENTATION.md: full implementation details and architecture.
- notebooks/Group_5_Capstone_Full_Pipeline.ipynb: notebook version of the full flow.
- scripts/run_capstone_pipeline.py: script-first orchestration for reproducible runs.

## Intended Use Notice

This work is a capstone research implementation for decision support and governance study. It must not be used as an autonomous lending decision engine.
