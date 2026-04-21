# Group 5 Capstone Project

## Project Overview

Provide a concise summary of:
- The problem this project addresses
- The dataset(s) used
- The capstone objective
- The Responsible AI objective

Example prompt:
- What decision-support question are we trying to answer, and why does fairness/governance matter for this use case?

## Team Members

- Name 1
- Name 2
- Name 3
- Name 4
- Name 5

## Problem Statement

State the research or business problem clearly, including:
- Who is impacted
- What decision the model informs
- Why the problem is important

## Scope

In scope:
- Item 1
- Item 2
- Item 3

Out of scope:
- Item 1
- Item 2

## Repository Structure

- `data/raw/`: original source data
- `data/processed/`: final analysis-ready data
- `notebooks/`: exploratory analysis and modeling notebooks
- `src/`: reusable code for loading, features, models, and visualization
- `models/`: saved model artifacts
- `reports/figures/`: exported charts
- `reports/tables/`: exported tables
- `docs/`: notes, scope, and governance documents
- `references/`: citations and source materials
- `scripts/`: repeatable scripts
- `tests/`: validation and unit tests

## Quick Start

Document exact setup and run steps.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_capstone_pipeline.py
```

Optional run modes:

```bash
python scripts/run_capstone_pipeline.py --max-rows 300000
python scripts/run_capstone_pipeline.py --use-full-dataset
python scripts/run_capstone_pipeline.py --input-file data/processed/hmda_lar_2024_cleaned.csv
```

## Data Sources

List the datasets, where they came from, and any access constraints.

### HMDA LAR Dataset Loading (Document This Clearly)

Use the HMDA 2024 LAR file naming convention `hmda_lar_2024.*` in `data/raw/`.

- Raw zip input: `data/raw/hmda_lar_2024.zip` (optional)
- Supported extracted/raw formats: `.csv`, `.txt`, `.tsv`, `.psv`
- Processed output: `data/processed/hmda_lar_2024_cleaned.csv`

Recommended loading flow:

1. Check whether `hmda_lar_2024.zip` exists and unzip to `data/raw/`.
2. Detect the source file by extension from supported formats.
3. Infer delimiter before reading:
4. Start from extension defaults (`.csv` -> `,`, `.tsv` -> tab, `.psv` -> `|`, `.txt` -> `,`).
5. Validate/override using `csv.Sniffer` on a sample.
6. Load with `pandas.read_csv(..., sep=detected_delimiter)`.
7. Clean and save to `data/processed/`.

Why this matters:

- Prevents read failures when HMDA files are not comma-delimited.
- Makes the pipeline robust to common tabular text variants.

## Methodology

Document end-to-end approach:
- Target construction
- Feature preparation and preprocessing
- Model candidates
- Threshold tuning logic
- Evaluation metrics

Suggested metric block:
- AUC
- Accuracy
- Log loss
- F1
- TPR, TNR, FPR, FNR
- Positive prediction rate

## Responsible AI and Governance

### Fairness

Document:
- Protected/group attributes audited
- AIR, mean effect, standardized mean difference
- Group error rates (FPR/FNR)
- Any significance testing approach

### Robustness and Drift

Document:
- Drift indicators (for example PSI, KS)
- Stress testing approach (for example perturbation)
- Alert thresholds and interpretation guidance

### Transparency

Document explainability outputs (for example):
- Coefficient tables for linear models
- Permutation feature importance

### Governance Outputs

Document governance artifacts generated:
- Deployment recommendation memo
- Monitoring thresholds
- Known limitations and abuse controls

## Model Selection Logic

Explain how final model is chosen.

Example:
- Composite score combining test performance, overfit penalty, and fairness penalty.

## Outputs

List all expected outputs.

Example:
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

Document testing command and current scope.

Example:

```bash
pytest
```

Include:
- What is currently covered by tests
- What is not yet covered

## Project Management

### Collaboration Workflow

- Branching strategy
- Pull request expectations
- Commit message expectations

### Cadence

- Planning cadence
- Progress review cadence
- Demo/review cadence

### Roles and Ownership

- Data lead
- Modeling lead
- Responsible AI lead
- Documentation lead
- QA lead

### Risk Management

Track project risks such as:
- Data drift/schema changes
- Fairness degradation
- Performance instability
- Reproducibility gaps

## Reproducibility Checklist

- Random seeds set and documented
- Dependencies pinned/documented
- Scripted entry point available
- Output artifacts versioned and reviewable

## Documentation Map

- docs/TECHNICAL_PIPELINE_DOCUMENTATION.md
- notebooks/Group_5_Capstone_Full_Pipeline.ipynb
- scripts/run_capstone_pipeline.py

## Deliverables

- Final report
- Presentation slides
- Reproducible notebook or script
- Supporting figures and tables

## Notes

Use this template as the starting point for the final project README.
