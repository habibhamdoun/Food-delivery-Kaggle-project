from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import OUTPUTS_DIR, TARGET_COLUMN
from src.data_utils import load_train_data
from src.features import build_features


def save_csv(df: pd.DataFrame, name: str) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / name
    df.to_csv(path, index=False)
    return path


def main() -> None:
    train_df = load_train_data()
    feature_df = build_features(train_df)

    print("Starting EDA summary generation...")
    print(f"Rows: {train_df.shape[0]}")
    print(f"Columns: {train_df.shape[1]}")
    print(f"Target mean: {train_df[TARGET_COLUMN].mean():.4f}")

    missing_summary = (
        train_df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing_summary["missing_fraction"] = missing_summary["missing_count"] / len(train_df)

    numeric_summary = train_df.describe(include=["number"]).transpose().reset_index()
    numeric_summary = numeric_summary.rename(columns={"index": "column"})

    categorical_cols = [col for col in train_df.columns if train_df[col].dtype == "object"]
    category_outputs: list[Path] = []
    for col in categorical_cols:
        summary = (
            train_df.groupby(col, dropna=False)[TARGET_COLUMN]
            .agg(["count", "mean"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        summary = summary.rename(columns={"mean": "target_rate"})
        category_outputs.append(save_csv(summary, f"eda_target_by_{col}.csv"))

    engineered_numeric_cols = [
        col for col in feature_df.columns if col != TARGET_COLUMN and pd.api.types.is_numeric_dtype(feature_df[col])
    ]
    correlation_summary = (
        feature_df[engineered_numeric_cols + [TARGET_COLUMN]]
        .corr(numeric_only=True)[TARGET_COLUMN]
        .drop(labels=[TARGET_COLUMN])
        .abs()
        .sort_values(ascending=False)
        .rename("abs_target_correlation")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    missing_path = save_csv(missing_summary.sort_values("missing_fraction", ascending=False), "eda_missing_summary.csv")
    numeric_path = save_csv(numeric_summary, "eda_numeric_summary.csv")
    corr_path = save_csv(correlation_summary, "eda_feature_target_correlation.csv")

    print("Saved EDA files:")
    print(f"  Missing summary: {missing_path}")
    print(f"  Numeric summary: {numeric_path}")
    print(f"  Feature correlation summary: {corr_path}")
    for path in category_outputs:
        print(f"  Target by category: {path}")


if __name__ == "__main__":
    main()
