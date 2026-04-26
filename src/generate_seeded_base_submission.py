from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from src.baseline import (
    fit_catboost_model,
    fit_lightgbm_model,
    predict_with_catboost,
    predict_with_lightgbm,
)
from src.config import ID_COLUMN, OUTPUTS_DIR, SUBMISSIONS_DIR, TARGET_COLUMN
from src.data_utils import load_test_data, load_train_data
from src.features import build_features_v1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["catboost", "lightgbm"],
        required=True,
        help="Base model family to train on the frozen v1 feature set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train and test data...")
    train_df = load_train_data()
    test_df = load_test_data()

    if args.model == "catboost":
        print(f"Training CatBoost on frozen v1 features with seed {args.seed}...")
        model, train_features, categorical_columns = fit_catboost_model(
            train_df,
            random_state=args.seed,
            feature_builder=build_features_v1,
        )
        test_predictions = predict_with_catboost(
            model=model,
            test_df=test_df,
            train_feature_frame=train_features,
            categorical_columns=categorical_columns,
            feature_builder=build_features_v1,
        )
    else:
        print(f"Training LightGBM on frozen v1 features with seed {args.seed}...")
        model, train_features, categorical_columns = fit_lightgbm_model(
            train_df,
            random_state=args.seed,
            feature_builder=build_features_v1,
        )
        test_predictions = predict_with_lightgbm(
            model=model,
            test_df=test_df,
            train_feature_frame=train_features,
            categorical_columns=categorical_columns,
            feature_builder=build_features_v1,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = SUBMISSIONS_DIR / f"{args.model}_seed{args.seed}_v1_submission_{timestamp}.csv"
    submission_df = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN],
            TARGET_COLUMN: test_predictions,
        }
    )
    submission_df.to_csv(submission_path, index=False)

    if hasattr(model, "get_feature_importance"):
        importance_values = model.get_feature_importance()
    else:
        importance_values = model.feature_importances_

    importance_path = OUTPUTS_DIR / f"{args.model}_seed{args.seed}_v1_feature_importance_{timestamp}.csv"
    pd.DataFrame(
        {
            "feature": train_features.columns,
            "importance": importance_values,
        }
    ).sort_values("importance", ascending=False).to_csv(importance_path, index=False)

    print(f"Saved seeded base submission to: {submission_path}")
    print(f"Saved feature importance to: {importance_path}")
    print("Submission preview:")
    print(submission_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
