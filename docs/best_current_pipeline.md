# Best Current Pipeline

## Best Public Kaggle Score

- Best public score: `0.97196`
- Best submission file: `submissions/blend_catboost_lightgbm_20260415_004218.csv`
- Best approach: 50/50 average of CatBoost and LightGBM probability predictions

## What Worked Best

The strongest submission so far did not come from a single model. It came from blending two strong tabular models:

- CatBoost
- LightGBM

Each model was trained on the original engineered feature set used in the first successful rounds. Averaging their probabilities equally produced a better Kaggle score than either model alone.

## What Did Not Beat It

- LightGBM alone improved over CatBoost alone, but not over the blend
- Weighted blends like 40/60 and 55/45 were slightly worse than 50/50
- A refreshed feature set improved local validation but did not improve Kaggle score
- A tuned LightGBM configuration performed very poorly on Kaggle and should not be reused

## Current Recommendation

For the final competition strategy, treat the following as the main model unless a later experiment clearly beats it:

- `blend_catboost_lightgbm_20260415_004218.csv`

If discussing the current best method in the report or presentation, describe it as:

"An ensemble that averages CatBoost and LightGBM probability predictions, which outperformed either individual model on the Kaggle public leaderboard."

## Suggested Report Framing

1. Start with single-model baselines.
2. Show CatBoost and LightGBM as the strongest individual models.
3. Explain that ensembling improved leaderboard performance.
4. Note that not every local validation improvement generalized to Kaggle.
5. Justify the final choice as the most reliable leaderboard performer.
