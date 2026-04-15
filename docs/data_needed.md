# Data Needed To Continue

Place the Kaggle competition files in the `data/` folder with names close to:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

The starter scripts are already prepared for those filenames.

## First Commands To Run

From the project root:

```powershell
python -m src.run_data_audit
python -m src.run_eda_summary
python -m src.train_baseline
python -m src.generate_submission
```

## What These Scripts Do

- `run_data_audit.py`: checks schema, prints target balance, and saves missing-value summary
- `run_eda_summary.py`: creates CSV summaries for missing values, numeric columns, category target rates, and engineered feature correlations
- `train_baseline.py`: compares a logistic baseline and a CatBoost baseline and reports fold AUC
- `generate_submission.py`: trains the current best model on the full training set and saves a Kaggle-ready submission CSV
