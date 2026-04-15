from __future__ import annotations

import pandas as pd

from src.baseline import (
    evaluate_catboost_baseline,
    evaluate_lightgbm_baseline,
    evaluate_logistic_baseline,
)
from src.config import OUTPUTS_DIR
from src.data_utils import load_train_data


def print_result(model_name: str, fold_scores: list[float], mean_auc: float) -> None:
    print(f"{model_name} fold ROC-AUC scores:")
    for idx, score in enumerate(fold_scores, start=1):
        print(f"  Fold {idx}: {score:.5f}")
    print(f"{model_name} mean ROC-AUC: {mean_auc:.5f}")
    print()


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_train_data()

    logistic_result = evaluate_logistic_baseline(train_df)
    print_result("Logistic baseline", logistic_result.fold_scores, logistic_result.mean_auc)

    catboost_result = evaluate_catboost_baseline(train_df)
    print_result("CatBoost baseline", catboost_result.fold_scores, catboost_result.mean_auc)

    lightgbm_result = evaluate_lightgbm_baseline(train_df)
    print_result("LightGBM baseline", lightgbm_result.fold_scores, lightgbm_result.mean_auc)

    rows: list[dict[str, float | int | str]] = []
    for model_name, result in [
        ("Logistic baseline", logistic_result),
        ("CatBoost baseline", catboost_result),
        ("LightGBM baseline", lightgbm_result),
    ]:
        for fold_idx, score in enumerate(result.fold_scores, start=1):
            rows.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "roc_auc": score,
                    "mean_roc_auc": result.mean_auc,
                }
            )

    results_df = pd.DataFrame(rows)
    csv_path = OUTPUTS_DIR / "baseline_results.csv"
    txt_path = OUTPUTS_DIR / "baseline_results.txt"
    results_df.to_csv(csv_path, index=False)

    with txt_path.open("w", encoding="utf-8") as handle:
        for model_name, result in [
            ("Logistic baseline", logistic_result),
            ("CatBoost baseline", catboost_result),
            ("LightGBM baseline", lightgbm_result),
        ]:
            handle.write(f"{model_name}\n")
            for fold_idx, score in enumerate(result.fold_scores, start=1):
                handle.write(f"  Fold {fold_idx}: {score:.5f}\n")
            handle.write(f"  Mean ROC-AUC: {result.mean_auc:.5f}\n\n")

    print(f"Saved baseline results to: {csv_path}")
    print(f"Saved baseline summary to: {txt_path}")


if __name__ == "__main__":
    main()
