# COE546/746 Food Delivery Order Prediction

This workspace is organized around the COE546/746 project on predicting whether a user completes an order during a food delivery app session.

## Objective

Build a machine learning pipeline that predicts `order_placed` for unseen sessions and produces Kaggle-ready probability submissions in the format:

```csv
id,order_placed
123,0.8421
```

The competition metric is AUC, so every experiment in this project should be evaluated primarily with ROC-AUC on local validation.

## Folder Structure

- `data/`: Kaggle files such as train, test, and sample submission
- `notebooks/`: exploratory and modeling notebooks
- `src/`: reusable Python modules for preprocessing, features, and modeling
- `outputs/`: figures, tables, and saved artifacts
- `submissions/`: generated Kaggle submission CSV files
- `docs/`: project planning notes, report outline, and presentation notes

## Immediate Next Steps

1. Download the Kaggle competition files into `data/`.
2. Open the starter notebook and confirm the dataset columns and target balance.
3. Build the first validation baseline using stratified cross-validation and ROC-AUC.
4. Track every experiment in `docs/experiment_log.md`.

## Modeling Priorities

- Start with trustworthy validation before aggressive tuning.
- Use timestamp-derived, cart-related, and promotion-related features early.
- Compare a simple interpretable baseline with stronger gradient boosting models.
- Prefer clean, reproducible experiments over leaderboard guessing.

## Deliverables

- A documented notebook that runs end-to-end
- A written report explaining the workflow and choices
- A presentation summarizing insights and final results
- A final Kaggle submission before the competition deadline
