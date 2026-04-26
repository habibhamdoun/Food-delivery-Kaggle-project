from __future__ import annotations

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


def percentile_normalize(series: pd.Series) -> pd.Series:
    if len(series) <= 1:
        return pd.Series([0.5] * len(series), index=series.index, dtype=float)
    ranks = series.rank(method="average")
    normalized = (ranks - 1) / (len(series) - 1)
    return normalized.astype(float)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train and test data...")
    train_df = load_train_data()
    test_df = load_test_data()

    print("Training final CatBoost model on frozen v1 features...")
    catboost_model, catboost_train_features, catboost_categorical_columns = fit_catboost_model(
        train_df,
        feature_builder=build_features_v1,
    )
    catboost_predictions = predict_with_catboost(
        model=catboost_model,
        test_df=test_df,
        train_feature_frame=catboost_train_features,
        categorical_columns=catboost_categorical_columns,
        feature_builder=build_features_v1,
    )

    print("Training final LightGBM model on frozen v1 features...")
    lightgbm_model, lightgbm_train_features, lightgbm_categorical_columns = fit_lightgbm_model(
        train_df,
        feature_builder=build_features_v1,
    )
    lightgbm_predictions = predict_with_lightgbm(
        model=lightgbm_model,
        test_df=test_df,
        train_feature_frame=lightgbm_train_features,
        categorical_columns=lightgbm_categorical_columns,
        feature_builder=build_features_v1,
    )

    print("Rank-averaging predictions and normalizing to probability-like scores...")
    prediction_frame = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN],
            "catboost_prediction": catboost_predictions,
            "lightgbm_prediction": lightgbm_predictions,
        }
    )
    prediction_frame["catboost_rank_score"] = percentile_normalize(prediction_frame["catboost_prediction"])
    prediction_frame["lightgbm_rank_score"] = percentile_normalize(prediction_frame["lightgbm_prediction"])
    prediction_frame[TARGET_COLUMN] = (
        0.5 * prediction_frame["catboost_rank_score"]
        + 0.5 * prediction_frame["lightgbm_rank_score"]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_details_path = OUTPUTS_DIR / f"final_pipeline_prediction_details_{timestamp}.csv"
    prediction_frame.to_csv(prediction_details_path, index=False)

    submission_df = prediction_frame[[ID_COLUMN, TARGET_COLUMN]].copy()
    submission_path = SUBMISSIONS_DIR / f"final_rank_ensemble_submission_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Saved final reproducible submission to: {submission_path}")
    print(f"Saved supporting prediction details to: {prediction_details_path}")
    print("Submission preview:")
    print(submission_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
