# Best Current Pipeline

## Best Public Kaggle Score

- Best public score: `0.97261`
- Best submission file: `submissions/rank_blend_cb_0.50_lgbm_0.50_20260426_042932.csv`
- Best approach: final rank-average blend of the two top-performing target-encoding blend winners

## Current Best Competition Pipeline

The current leaderboard-best workflow is now a three-step competition strategy:

1. Generate the target-encoded LightGBM submission
2. Rank-blend it with the previous best rank-ensemble submission using slightly higher weight on the target-encoded model
3. Rank-blend the two strongest target-encoding blend winners

Relevant artifacts:

- Base target-encoded model:
  - `submissions/target_encoded_lightgbm_submission_20260425_023540.csv`
- Previous best rank-ensemble model:
  - `submissions/rank_blend_cb_0.50_lgbm_0.50_20260419_142129.csv`
- First target-encoding blend winner:
  - `submissions/rank_blend_cb_0.45_lgbm_0.55_20260425_032637.csv`
- Final best blended result:
  - `submissions/rank_blend_cb_0.50_lgbm_0.50_20260426_042932.csv`

Scripts involved:

- `python -m src.target_encoding_submission`
- `python -m src.rank_blend_submissions --catboost-file rank_blend_cb_0.50_lgbm_0.50_20260419_142129.csv --lightgbm-file target_encoded_lightgbm_submission_20260425_023540.csv --catboost-weight 0.45 --lightgbm-weight 0.55`
- `python -m src.rank_blend_submissions --catboost-file rank_blend_cb_0.45_lgbm_0.55_20260425_032637.csv --lightgbm-file rank_blend_cb_0.45_lgbm_0.55_20260426_042749.csv --catboost-weight 0.50 --lightgbm-weight 0.50`

## Reproducible Single-Model Improvement

The project still includes a reproducible target-encoding pipeline script:

- `python -m src.target_encoding_submission`

This script:

- starts from the frozen v1 numeric feature set
- adds leakage-safe target encodings for high-value categorical columns
- trains LightGBM with 5-fold out-of-fold validation logic
- saves a Kaggle-ready single-model submission

Most recent reproducible submission artifact:

- `submissions/target_encoded_lightgbm_submission_20260425_023540.csv`

## What Worked Best

The strongest submission so far is now a final rank-average blend between:

- the two strongest target-encoding blend winners

The earlier ensemble remained strong, and the target-encoded LightGBM finally surpassed it by capturing category-level conversion behavior more effectively. Blending those winners produced another improvement, and blending the top two target-encoding blend winners produced the final best score.

## What Did Not Beat It

- A simple LightGBM alone improved over CatBoost alone, but not over the old ensemble
- Probability-average weighted blends like 40/60 and 55/45 were slightly worse than the simple 50/50 probability blend
- Rank averaging improved over probability blending and remained a strong fallback
- A refreshed feature set improved local validation but did not improve Kaggle score
- A tuned LightGBM configuration performed very poorly on Kaggle and should not be reused

## Current Recommendation

For the final competition strategy, treat the following as the main model unless a later experiment clearly beats it:

- `rank_blend_cb_0.50_lgbm_0.50_20260426_042932.csv`

If discussing the current best method in the report or presentation, describe it as:

"A final rank-average ensemble that combines the two top-performing target-encoding blend winners. This produced the strongest public leaderboard score."

## Suggested Report Framing

1. Start with single-model baselines.
2. Show CatBoost and LightGBM as the strongest individual models.
3. Explain that ensembling improved leaderboard performance, and rank averaging improved it further.
4. Show that leakage-safe target encoding produced the next major jump.
5. Explain that blending the two strongest winners produced the next best score.
6. Explain that rank-blending the two top target-encoding winners produced the final best score.
7. Note that not every local validation improvement generalized to Kaggle.
8. Justify the final choice as the strongest leaderboard performer.
