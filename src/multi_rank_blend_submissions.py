from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

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
        "--files",
        nargs="+",
        required=True,
        help="Submission filenames inside the submissions folder.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        required=True,
        help="Weights matching the provided submission files.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional short tag to distinguish output filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    if len(args.files) != len(args.weights):
        raise ValueError("The number of files must match the number of weights.")

    total_weight = sum(args.weights)
    if total_weight <= 0:
        raise ValueError("The sum of weights must be positive.")

    normalized_weights = [weight / total_weight for weight in args.weights]

    merged: pd.DataFrame | None = None
    labels: list[str] = []

    for idx, (filename, weight) in enumerate(zip(args.files, normalized_weights), start=1):
        path = SUBMISSIONS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing submission file: {path}")

        label = Path(filename).stem[:24].replace("submission", "sub")
        labels.append(label)
        df = load_submission(path).rename(columns={TARGET_COLUMN: f"score_{idx}"})

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=ID_COLUMN, how="inner")

    assert merged is not None

    blended_rank_score = pd.Series(0.0, index=merged.index)
    for idx, weight in enumerate(normalized_weights, start=1):
        rank_col = f"rank_{idx}"
        score_col = f"score_{idx}"
        merged[rank_col] = merged[score_col].rank(method="average")
        blended_rank_score = blended_rank_score + weight * merged[rank_col]

    if len(merged) <= 1:
        merged[TARGET_COLUMN] = 0.5
    else:
        merged[TARGET_COLUMN] = (blended_rank_score - 1) / (len(merged) - 1)

    output = merged[[ID_COLUMN, TARGET_COLUMN]].copy()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weight_tag = "_".join(f"w{weight:.2f}" for weight in normalized_weights)
    tag_part = f"{args.tag}_" if args.tag else ""
    output_path = SUBMISSIONS_DIR / f"multi_rank_blend_{tag_part}{weight_tag}_{timestamp}.csv"
    output.to_csv(output_path, index=False)

    print(f"Saved multi-rank blended submission to: {output_path}")
    print("Files and normalized weights used:")
    for filename, weight in zip(args.files, normalized_weights):
        print(f"  {filename}: {weight:.2f}")
    print("Final output was normalized to the [0, 1] range.")
    print("Submission preview:")
    print(output.head().to_string(index=False))


if __name__ == "__main__":
    main()
