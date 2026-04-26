from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ID_COLUMN, SUBMISSIONS_DIR, TARGET_COLUMN


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {ID_COLUMN, TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Submission file {path} is missing columns: {sorted(missing)}")
    return df[[ID_COLUMN, TARGET_COLUMN]].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catboost-file",
        type=str,
        default="catboost_submission_20260413_172709.csv",
        help="CatBoost submission filename inside the submissions folder.",
    )
    parser.add_argument(
        "--lightgbm-file",
        type=str,
        default="lightgbm_submission_20260413_181714.csv",
        help="LightGBM submission filename inside the submissions folder.",
    )
    parser.add_argument(
        "--catboost-weight",
        type=float,
        default=0.5,
        help="Weight for CatBoost logits.",
    )
    parser.add_argument(
        "--lightgbm-weight",
        type=float,
        default=0.5,
        help="Weight for LightGBM logits.",
    )
    return parser.parse_args()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def main() -> None:
    args = parse_args()
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    catboost_path = SUBMISSIONS_DIR / args.catboost_file
    lightgbm_path = SUBMISSIONS_DIR / args.lightgbm_file

    if not catboost_path.exists():
        raise FileNotFoundError(f"Missing CatBoost submission file: {catboost_path}")
    if not lightgbm_path.exists():
        raise FileNotFoundError(f"Missing LightGBM submission file: {lightgbm_path}")

    catboost_df = load_submission(catboost_path)
    lightgbm_df = load_submission(lightgbm_path)

    merged = catboost_df.merge(
        lightgbm_df,
        on=ID_COLUMN,
        suffixes=("_catboost", "_lightgbm"),
        how="inner",
    )

    total_weight = args.catboost_weight + args.lightgbm_weight
    if total_weight <= 0:
        raise ValueError("The sum of weights must be positive.")

    catboost_weight = args.catboost_weight / total_weight
    lightgbm_weight = args.lightgbm_weight / total_weight

    eps = 1e-6
    catboost_probs = merged[f"{TARGET_COLUMN}_catboost"].clip(eps, 1 - eps)
    lightgbm_probs = merged[f"{TARGET_COLUMN}_lightgbm"].clip(eps, 1 - eps)

    catboost_logits = np.log(catboost_probs / (1 - catboost_probs))
    lightgbm_logits = np.log(lightgbm_probs / (1 - lightgbm_probs))

    blended_logits = catboost_weight * catboost_logits + lightgbm_weight * lightgbm_logits
    merged[TARGET_COLUMN] = sigmoid(blended_logits.to_numpy())

    output = merged[[ID_COLUMN, TARGET_COLUMN]].copy()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = SUBMISSIONS_DIR / (
        f"logit_blend_cb_{catboost_weight:.2f}_lgbm_{lightgbm_weight:.2f}_{timestamp}.csv"
    )
    output.to_csv(output_path, index=False)

    print(f"Saved logit-blended submission to: {output_path}")
    print(
        "Logit blend used weighted log-odds with "
        f"CatBoost={catboost_weight:.2f} and LightGBM={lightgbm_weight:.2f}."
    )
    print("Submission preview:")
    print(output.head().to_string(index=False))


if __name__ == "__main__":
    main()
