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
python -m src.train_baseline
```

## What These Scripts Do

- `run_data_audit.py`: checks schema, prints target balance, and saves missing-value summary
- `train_baseline.py`: builds a first logistic regression baseline with engineered features and reports fold AUC
