# Home Credit Feature Pipeline

A leakage-safe, tested, documented feature pipeline for credit default prediction on the Home Credit dataset.

> Work in progress — results, approach and limitations will be added as stages are completed.

## Requirements
- Python 3.13.14
- Windows / PowerShell (commands below); on Linux/macOS use `source .venv/bin/activate`

## How to run
1. Clone the repo:
```
   git clone https://github.com/Meck37/home-credit-feature-pipeline.git
   cd home-credit-feature-pipeline
```
2. Create and activate a virtual environment:
```
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
```
3. Install pinned dependencies:
```
   python -m pip install -r requirements.txt
```
4. Download the data (see below).
5. Run the tests:
```
   pytest
```

## Getting the data
The dataset is not committed to this repo because Kaggle's competition rules don't allow redistribution, and the files are several GB.

1. Accept the competition rules at https://www.kaggle.com/competitions/home-credit-default-risk
2. Download from the Data tab and place these files in `data/raw/`:
   - `application_train.csv`
   - `bureau.csv`
   - `previous_application.csv`
   - `installments_payments.csv`
   - `HomeCredit_columns_description.csv`

## Repo map
- `src/` — feature logic, split, evaluation (all reusable code lives here)
- `tests/` — unit tests for feature functions
- `notebooks/` — exploration and reporting only; they import from `src/`
- `docs/` — feature catalog
- `reports/` — results tables and leakage audit
- `DECISIONS.md` — reasoning log for every stage