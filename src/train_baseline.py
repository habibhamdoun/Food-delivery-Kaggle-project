from __future__ import annotations

from src.baseline import evaluate_logistic_baseline
from src.data_utils import load_train_data


def main() -> None:
    train_df = load_train_data()
    result = evaluate_logistic_baseline(train_df)

    print("Fold ROC-AUC scores:")
    for idx, score in enumerate(result.fold_scores, start=1):
        print(f"  Fold {idx}: {score:.5f}")

    print(f"Mean ROC-AUC: {result.mean_auc:.5f}")


if __name__ == "__main__":
    main()
