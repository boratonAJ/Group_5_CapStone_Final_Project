# Data Directory

## Source

- **Dataset:** 2024 HMDA Loan/Application Records (LAR)  
- **Source:** CFPB HMDA Data Browser — https://ffiec.cfpb.gov/data-browser/  
- **Format:** Pipe-delimited text (99 columns, header row)  
- **Filename in raw/:** `2024_lar.txt` (inside `2024_lar.zip`)
- **Snapshot date:** [INSERT DATE OF DOWNLOAD]

## File Hash (SHA-256)

```
3f1578c027b1ae388e944492ac7b24d73a4deeb5da73ff731bebf34358a256a9
```

Verify with:
```python
import hashlib
with open("data/raw/2024_lar.zip", "rb") as f:
    print(hashlib.sha256(f.read()).hexdigest())
```

## Row Counts (populated after 01_data_prep.ipynb)

| Step | Row Count |
|---|---|
| Raw LAR rows | [INSERT] |
| action_taken in {1,2,3} | [INSERT] |
| After leakage feature removal | [INSERT] (same rows, fewer columns) |
| Final modeling frame | [INSERT] |
| Train set (70%) | [INSERT] |
| Validation set (15%) | [INSERT] |
| Test set (15%) | [INSERT] |
| Geographic holdout | [INSERT] |

## Label Distribution

| Label | action_taken values | Count | % |
|---|---|---|---|
| y=1 (originated/approved) | 1, 2 | [INSERT] | [INSERT] |
| y=0 (denied) | 3 | [INSERT] | [INSERT] |

## Processed Files (populated after 01_data_prep.ipynb)

| File | Description |
|---|---|
| `processed/modeling_frame.parquet` | Full labeled dataset after leakage removal |
| `processed/train.parquet` | Training split |
| `processed/val.parquet` | Validation split |
| `processed/test.parquet` | Test split |
| `processed/geo_holdout.parquet` | Geographic holdout for drift testing |

## Setup Instructions

1. Download 2024 HMDA LAR from CFPB HMDA Data Browser.  
2. Place `2024_lar.zip` in this repository's root (same level as `hmda-capstone/`) or set `DATA_PATH` in `notebooks/01_data_prep.ipynb`.  
3. Verify SHA-256 hash matches above.  
4. Run `notebooks/01_data_prep.ipynb` to generate all processed files.

## Privacy Note

The raw HMDA LAR is public data published by CFPB. It contains aggregated lender-reported fields; individual applicants are not directly identified. However, the data contains sensitive financial and demographic attributes and should be handled in compliance with institutional data governance policies.
