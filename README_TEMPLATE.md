# Group 5 Capstone Project

## Project Overview

Provide a short description of the problem, the dataset, and the capstone goal.

## Team Members

- Name 1
- Name 2
- Name 3
- Name 4
- Name 5

## Repository Structure

- `data/raw/`: original source data
- `data/interim/`: temporary cleaned data
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

## Problem Statement

State the research question or business problem the capstone addresses.

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

Describe the planned modeling, evaluation, and responsible AI checks.

## Deliverables

- Final report
- Presentation slides
- Reproducible notebook or script
- Supporting figures and tables

## Notes

Use this template as the starting point for the final project README.
