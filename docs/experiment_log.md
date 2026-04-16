# Experiment Log

Use this file to keep the project disciplined. Each major run should be recorded here.

| Date | Experiment ID | Features | Model | Validation Setup | CV AUC | Kaggle Public Score | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-11 | baseline-plan | None yet | None yet | None yet | TBD | TBD | Project scaffold created and execution plan defined |
| 2026-04-13 | v1-catboost | Time, cart, and promotion engineered features | CatBoost | 5-fold stratified CV | 0.97070 | 0.97136 | First Kaggle submission; reached rank 5 at submission time |
| 2026-04-13 | v2-lightgbm | Time, cart, and promotion engineered features | LightGBM | 5-fold stratified CV | 0.97129 | 0.97144 | Second Kaggle submission; improved over CatBoost and reached rank 7 on April 15, 2026 |
| 2026-04-15 | v3-blend-50-50 | Same feature set as v1 and v2 | 50/50 CatBoost + LightGBM blend | Blend of submitted prediction files | N/A | 0.97196 | Best submission so far; tied 2nd by score and shown 3rd on leaderboard at submission time |
| 2026-04-15 | v3b-blend-40-60 | Same feature set as v1 and v2 | 40% CatBoost + 60% LightGBM blend | Blend of submitted prediction files | N/A | 0.97194 | Slightly worse than the 50/50 blend |
| 2026-04-15 | v3c-blend-55-45 | Same feature set as v1 and v2 | 55% CatBoost + 45% LightGBM blend | Blend of submitted prediction files | N/A | 0.97195 | Very close, but still worse than the 50/50 blend |
| 2026-04-16 | v4-feature-refresh-blend | Expanded session and promotion interaction features | New 50/50 CatBoost + LightGBM blend | Local CV improved, then blended fresh predictions | 0.97155 (LightGBM local) | 0.97184 | Feature refresh helped local validation but did not beat the old best Kaggle blend |
| 2026-04-16 | v5-tuned-lightgbm | Expanded feature set plus tuned LightGBM hyperparameters | Tuned LightGBM | 3-fold tuning search, best local mean 0.97166 | 0.97166 | 0.91710 | Major Kaggle failure; discard this configuration for future submissions |
