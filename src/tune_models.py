from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.baseline import evaluate_catboost_baseline, evaluate_lightgbm_baseline
from src.config import OUTPUTS_DIR
from src.data_utils import load_train_data


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_train_data()

    experiments: list[tuple[str, dict]] = [
        (
            "lightgbm",
            {
                "n_estimators": 700,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        ),
        (
            "lightgbm",
            {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "subsample": 0.85,
                "colsample_bytree": 0.8,
            },
        ),
        (
            "lightgbm",
            {
                "n_estimators": 600,
                "learning_rate": 0.04,
                "num_leaves": 47,
                "subsample": 0.8,
                "colsample_bytree": 0.9,
            },
        ),
        (
            "catboost",
            {
                "iterations": 600,
                "learning_rate": 0.03,
                "depth": 6,
            },
        ),
        (
            "catboost",
            {
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 7,
            },
        ),
        (
            "catboost",
            {
                "iterations": 700,
                "learning_rate": 0.03,
                "depth": 5,
            },
        ),
    ]

    rows: list[dict] = []
    for idx, (model_name, params) in enumerate(experiments, start=1):
        print(f"Running experiment {idx}/{len(experiments)}: {model_name} with {params}")
        if model_name == "lightgbm":
            result = evaluate_lightgbm_baseline(
                train_df=train_df,
                n_splits=3,
                model_params=params,
            )
        else:
            result = evaluate_catboost_baseline(
                train_df=train_df,
                n_splits=3,
                model_params=params,
            )

        rows.append(
            {
                "model": model_name,
                "params_json": json.dumps(params, sort_keys=True),
                "mean_roc_auc": result.mean_auc,
                "fold_1": result.fold_scores[0],
                "fold_2": result.fold_scores[1],
                "fold_3": result.fold_scores[2],
            }
        )
        print(f"  Mean ROC-AUC: {result.mean_auc:.5f}")

    results_df = pd.DataFrame(rows).sort_values("mean_roc_auc", ascending=False)
    output_path = OUTPUTS_DIR / "tuning_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"Saved tuning results to: {output_path}")
    print("Top experiments:")
    print(results_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
