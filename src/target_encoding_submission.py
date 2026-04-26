from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import ID_COLUMN, OUTPUTS_DIR, SUBMISSIONS_DIR, TARGET_COLUMN
from src.data_utils import load_test_data, load_train_data
from src.features import build_features_v1


TARGET_ENCODE_COLUMNS = [
    "f7",
    "f9",
    "f12",
    "f17",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoothing",
        type=float,
        default=20.0,
        help="Smoothing strength for target encoding.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag for output filenames.",
    )
    return parser.parse_args()


def add_combo_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if {"f12", "f17"}.issubset(frame.columns):
        frame["te_combo_promo_response"] = (
            frame["f12"].fillna("missing").astype(str) + "_" + frame["f17"].fillna("missing").astype(str)
        )
    if {"f7", "f17"}.issubset(frame.columns):
        frame["te_combo_action_response"] = (
            frame["f7"].fillna("missing").astype(str) + "_" + frame["f17"].fillna("missing").astype(str)
        )
    return frame


def target_encode_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    columns: list[str],
    n_splits: int = 5,
    random_state: int = 42,
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_encoded = pd.DataFrame(index=train_df.index)
    test_encoded = pd.DataFrame(index=test_df.index)
    global_mean = float(y.mean())

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for col in columns:
        train_encoded[f"{col}_te"] = np.nan
        test_fold_values: list[np.ndarray] = []

        for train_idx, valid_idx in cv.split(train_df, y):
            x_tr = train_df.iloc[train_idx][col].fillna("missing").astype(str)
            y_tr = y.iloc[train_idx]
            x_val = train_df.iloc[valid_idx][col].fillna("missing").astype(str)

            stats = (
                pd.DataFrame({"key": x_tr, "target": y_tr})
                .groupby("key")["target"]
                .agg(["mean", "count"])
            )
            smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
            mapped = x_val.map(smooth).fillna(global_mean)
            train_encoded.iloc[valid_idx, train_encoded.columns.get_loc(f"{col}_te")] = mapped.to_numpy()

            x_test = test_df[col].fillna("missing").astype(str)
            test_fold_values.append(x_test.map(smooth).fillna(global_mean).to_numpy())

        test_encoded[f"{col}_te"] = np.mean(np.vstack(test_fold_values), axis=0)

    return train_encoded, test_encoded


def prepare_lightgbm_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    smoothing: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_train = build_features_v1(train_df)
    base_test = build_features_v1(test_df)

    numeric_columns = base_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != TARGET_COLUMN]
    train_numeric = base_train[numeric_columns].copy()
    test_numeric = base_test[numeric_columns].copy()

    train_cats = add_combo_columns(train_df)
    test_cats = add_combo_columns(test_df)
    te_columns = [col for col in TARGET_ENCODE_COLUMNS + ["te_combo_promo_response", "te_combo_action_response"] if col in train_cats.columns]

    train_target = train_df[TARGET_COLUMN].astype(int)
    te_train, te_test = target_encode_train_test(
        train_cats,
        test_cats,
        train_target,
        te_columns,
        smoothing=smoothing,
    )

    X_train = pd.concat([train_numeric, te_train], axis=1)
    X_test = pd.concat([test_numeric, te_test], axis=1)

    for col in X_train.columns:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    return X_train, X_test


def main() -> None:
    args = parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train and test data...")
    train_df = load_train_data()
    test_df = load_test_data()
    y = train_df[TARGET_COLUMN].astype(int)

    print(f"Building target-encoded training and test frames with smoothing={args.smoothing}...")
    X_train, X_test = prepare_lightgbm_frames(train_df, test_df, smoothing=args.smoothing)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X_train), dtype=float)
    test_fold_preds: list[np.ndarray] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train, y), start=1):
        print(f"Training fold {fold_idx}/5...")
        model = LGBMClassifier(
            objective="binary",
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train.iloc[train_idx], y.iloc[train_idx])
        oof_pred[valid_idx] = model.predict_proba(X_train.iloc[valid_idx])[:, 1]
        test_fold_preds.append(model.predict_proba(X_test)[:, 1])

    oof_auc = roc_auc_score(y, oof_pred)
    print(f"OOF ROC-AUC: {oof_auc:.5f}")

    test_pred = np.mean(np.vstack(test_fold_preds), axis=0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"{args.tag}_" if args.tag else ""
    smoothing_tag = str(args.smoothing).replace(".", "p")
    submission_path = SUBMISSIONS_DIR / f"target_encoded_lightgbm_{tag_part}s{ smoothing_tag }_{timestamp}.csv"
    submission_df = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN],
            TARGET_COLUMN: test_pred,
        }
    )
    submission_df.to_csv(submission_path, index=False)

    oof_path = OUTPUTS_DIR / f"target_encoded_lightgbm_oof_{tag_part}s{ smoothing_tag }_{timestamp}.csv"
    pd.DataFrame({"oof_pred": oof_pred, TARGET_COLUMN: y}).to_csv(oof_path, index=False)

    print(f"Saved target-encoded LightGBM submission to: {submission_path}")
    print(f"Saved OOF details to: {oof_path}")
    print("Submission preview:")
    print(submission_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
