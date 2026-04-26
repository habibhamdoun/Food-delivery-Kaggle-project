from __future__ import annotations

import json

import pandas as pd

from src.baseline import evaluate_catboost_baseline
from src.config import OUTPUTS_DIR
from src.data_utils import load_train_data


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_train_data()

    experiments: list[dict] = [
        {
            "label": "catboost_baseline_reference",
            "params": {
                "iterations": 400,
                "learning_rate": 0.05,
                "depth": 6,
            },
        },
        {
            "label": "catboost_depth5_regularized",
            "params": {
                "iterations": 700,
                "learning_rate": 0.03,
                "depth": 5,
                "l2_leaf_reg": 5,
                "random_strength": 1,
            },
        },
        {
            "label": "catboost_depth6_regularized",
            "params": {
                "iterations": 700,
                "learning_rate": 0.03,
                "depth": 6,
                "l2_leaf_reg": 7,
                "random_strength": 1.5,
            },
        },
        {
            "label": "catboost_depth6_stable",
            "params": {
                "iterations": 550,
                "learning_rate": 0.04,
                "depth": 6,
                "l2_leaf_reg": 9,
                "random_strength": 2,
            },
        },
        {
            "label": "catboost_depth7_light_reg",
            "params": {
                "iterations": 500,
                "learning_rate": 0.04,
                "depth": 7,
                "l2_leaf_reg": 4,
                "random_strength": 1,
            },
        },
    ]

    rows: list[dict] = []
    for idx, experiment in enumerate(experiments, start=1):
        label = experiment["label"]
        params = experiment["params"]
        print(f"Running CatBoost experiment {idx}/{len(experiments)}: {label}")
        print(f"  Params: {params}")
        result = evaluate_catboost_baseline(
            train_df=train_df,
            n_splits=3,
            model_params=params,
        )
        rows.append(
            {
                "label": label,
                "params_json": json.dumps(params, sort_keys=True),
                "mean_roc_auc": result.mean_auc,
                "fold_1": result.fold_scores[0],
                "fold_2": result.fold_scores[1],
                "fold_3": result.fold_scores[2],
            }
        )
        print(f"  Mean ROC-AUC: {result.mean_auc:.5f}")

    results_df = pd.DataFrame(rows).sort_values("mean_roc_auc", ascending=False)
    output_path = OUTPUTS_DIR / "catboost_tuning_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"Saved CatBoost tuning results to: {output_path}")
    print("Top CatBoost experiments:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
