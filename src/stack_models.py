from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.baseline import (
    fit_catboost_model,
    fit_lightgbm_model,
    predict_with_catboost,
    predict_with_lightgbm,
)
from src.config import ID_COLUMN, OUTPUTS_DIR, SUBMISSIONS_DIR, TARGET_COLUMN
from src.data_utils import load_test_data, load_train_data
from src.features import build_features_v1


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train and test data...")
    train_df = load_train_data()
    test_df = load_test_data()
    y = train_df[TARGET_COLUMN].astype(int).to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_catboost = np.zeros(len(train_df), dtype=float)
    oof_lightgbm = np.zeros(len(train_df), dtype=float)
    test_catboost_folds: list[np.ndarray] = []
    test_lightgbm_folds: list[np.ndarray] = []
    fold_scores: list[dict[str, float | int]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(train_df, y), start=1):
        print(f"Training fold {fold_idx}/5...")
        fold_train = train_df.iloc[train_idx].copy()
        fold_valid = train_df.iloc[valid_idx].copy()

        catboost_model, catboost_train_features, catboost_categorical_columns = fit_catboost_model(
            fold_train,
            feature_builder=build_features_v1,
        )
        valid_catboost = predict_with_catboost(
            model=catboost_model,
            test_df=fold_valid,
            train_feature_frame=catboost_train_features,
            categorical_columns=catboost_categorical_columns,
            feature_builder=build_features_v1,
        )
        test_catboost = predict_with_catboost(
            model=catboost_model,
            test_df=test_df,
            train_feature_frame=catboost_train_features,
            categorical_columns=catboost_categorical_columns,
            feature_builder=build_features_v1,
        )

        lightgbm_model, lightgbm_train_features, lightgbm_categorical_columns = fit_lightgbm_model(
            fold_train,
            feature_builder=build_features_v1,
        )
        valid_lightgbm = predict_with_lightgbm(
            model=lightgbm_model,
            test_df=fold_valid,
            train_feature_frame=lightgbm_train_features,
            categorical_columns=lightgbm_categorical_columns,
            feature_builder=build_features_v1,
        )
        test_lightgbm = predict_with_lightgbm(
            model=lightgbm_model,
            test_df=test_df,
            train_feature_frame=lightgbm_train_features,
            categorical_columns=lightgbm_categorical_columns,
            feature_builder=build_features_v1,
        )

        oof_catboost[valid_idx] = valid_catboost
        oof_lightgbm[valid_idx] = valid_lightgbm
        test_catboost_folds.append(test_catboost)
        test_lightgbm_folds.append(test_lightgbm)

        fold_scores.append(
            {
                "fold": fold_idx,
                "catboost_auc": roc_auc_score(y[valid_idx], valid_catboost),
                "lightgbm_auc": roc_auc_score(y[valid_idx], valid_lightgbm),
            }
        )

    stack_train = pd.DataFrame(
        {
            "catboost_pred": oof_catboost,
            "lightgbm_pred": oof_lightgbm,
        }
    )
    stack_test = pd.DataFrame(
        {
            "catboost_pred": np.mean(test_catboost_folds, axis=0),
            "lightgbm_pred": np.mean(test_lightgbm_folds, axis=0),
        }
    )

    print("Training stacking meta-model...")
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(stack_train, y)
    oof_stack = meta_model.predict_proba(stack_train)[:, 1]
    test_stack = meta_model.predict_proba(stack_test)[:, 1]
    stack_auc = roc_auc_score(y, oof_stack)
    print(f"OOF stacking ROC-AUC: {stack_auc:.5f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = SUBMISSIONS_DIR / f"stacking_submission_{timestamp}.csv"
    submission_df = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN],
            TARGET_COLUMN: test_stack,
        }
    )
    submission_df.to_csv(submission_path, index=False)

    fold_scores_df = pd.DataFrame(fold_scores)
    scores_path = OUTPUTS_DIR / f"stacking_fold_scores_{timestamp}.csv"
    fold_scores_df.to_csv(scores_path, index=False)

    oof_details_path = OUTPUTS_DIR / f"stacking_oof_details_{timestamp}.csv"
    stack_train.assign(order_placed=y, stack_pred=oof_stack).to_csv(oof_details_path, index=False)

    print(f"Saved stacking submission to: {submission_path}")
    print(f"Saved fold-level scores to: {scores_path}")
    print(f"Saved OOF stacking details to: {oof_details_path}")
    print("Submission preview:")
    print(submission_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
