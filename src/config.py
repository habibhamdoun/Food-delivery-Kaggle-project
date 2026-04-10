from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

TRAIN_FILE_CANDIDATES = [
    "train.csv",
    "Train.csv",
    "training.csv",
]

TEST_FILE_CANDIDATES = [
    "test.csv",
    "Test.csv",
]

SAMPLE_SUBMISSION_CANDIDATES = [
    "sample_submission.csv",
    "SampleSubmission.csv",
    "submission.csv",
]

TARGET_COLUMN = "order_placed"
ID_COLUMN = "id"
