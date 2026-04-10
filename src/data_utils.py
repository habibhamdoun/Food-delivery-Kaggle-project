from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    DATA_DIR,
    SAMPLE_SUBMISSION_CANDIDATES,
    TEST_FILE_CANDIDATES,
    TRAIN_FILE_CANDIDATES,
)


def resolve_existing_file(candidates: list[str], base_dir: Path = DATA_DIR) -> Path:
    for name in candidates:
        path = base_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find any of these files in {base_dir}: {', '.join(candidates)}"
    )


def load_train_data() -> pd.DataFrame:
    path = resolve_existing_file(TRAIN_FILE_CANDIDATES)
    return pd.read_csv(path)


def load_test_data() -> pd.DataFrame:
    path = resolve_existing_file(TEST_FILE_CANDIDATES)
    return pd.read_csv(path)


def load_sample_submission() -> pd.DataFrame:
    path = resolve_existing_file(SAMPLE_SUBMISSION_CANDIDATES)
    return pd.read_csv(path)
