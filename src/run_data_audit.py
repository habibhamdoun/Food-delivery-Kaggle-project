from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import OUTPUTS_DIR, TARGET_COLUMN
from src.data_utils import load_test_data, load_train_data


def main() -> None:
    train_df = load_train_data()
    test_df = load_test_data()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Train columns:", list(train_df.columns))
    print("Test columns:", list(test_df.columns))

    if TARGET_COLUMN in train_df.columns:
        target_rate = train_df[TARGET_COLUMN].mean()
        print(f"Target mean ({TARGET_COLUMN}): {target_rate:.4f}")

    missing = (
        train_df.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_fraction")
        .reset_index()
        .rename(columns={"index": "column"})
    )

    output_path = OUTPUTS_DIR / "train_missing_summary.csv"
    missing.to_csv(output_path, index=False)
    print(f"Saved missing value summary to: {output_path}")


if __name__ == "__main__":
    main()
