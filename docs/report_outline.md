# Report Outline

## Title

User Behavior Prediction in Food Delivery Applications

## 1. Introduction

Explain the problem in simple terms:

- The goal is to predict whether a user will complete an order during a food delivery app session.
- The dataset contains session-level behavior, cart activity, timing information, and promotion interactions.
- The competition metric is ROC-AUC, so the model must rank positive sessions above negative ones as accurately as possible.

Points to include:

- Why this problem matters in a product context
- Why promotion and session behavior are likely useful predictors
- Why class imbalance makes the task more challenging

## 2. Dataset Description

Summarize the data:

- Training rows: `297,236`
- Test rows: `99,639`
- Training columns: `18`
- Test columns: `17`
- Target variable: `order_placed`

Mention important feature groups:

- Session timing: `f3`, `f4`, `f5`
- User and context information: `f6`, `f7`, `f9`, `f16`
- Cart behavior: `f10`, `f11`
- Promotion behavior: `f8`, `f12`, `f13`, `f14`, `f15`, `f17`

## 3. Exploratory Data Analysis

Describe the key findings from EDA:

- The target is highly imbalanced, with only about `2.91%` positive sessions.
- Several promotion-related columns have substantial missingness.
- Features related to cart value, session duration, and promotion qualification showed strong predictive value.

Good evidence to reference:

- [eda_missing_summary.csv](C:\Users\habib\Desktop\ML%20project\coe546-food-delivery\outputs\eda_missing_summary.csv)
- [eda_feature_target_correlation.csv](C:\Users\habib\Desktop\ML%20project\coe546-food-delivery\outputs\eda_feature_target_correlation.csv)

## 4. Data Preprocessing

Explain what was done before modeling:

- Timestamp columns were converted to datetime format.
- Missing values were handled differently depending on model type.
- Categorical values were encoded appropriately for each model.
- Identifier columns were handled carefully and later removed from modeling when they were found not to add reusable information.

Important observation:

- Every `id` appeared only once, so customer-history aggregation by `id` was not possible.

## 5. Feature Engineering

Explain that new features were derived from existing columns, not invented externally.

Examples to discuss:

- `session_duration_seconds`
- `active_span_seconds`
- `idle_gap_seconds`
- `avg_item_value`
- `declined_offer_ratio`
- `cart_to_min_order_gap`
- `meets_min_order`
- promotion response indicator columns
- interaction features such as action + promotion response

Also explain the motivation:

- raw timestamps are less useful than session duration
- raw promotion columns become more informative when converted into ratios and gap-based measures
- interaction features help the model capture different user response patterns

## 6. Modeling Approach

Present the progression clearly:

1. Logistic Regression baseline
2. CatBoost baseline
3. LightGBM baseline
4. CatBoost + LightGBM ensemble

Key local results to mention:

- Logistic baseline: around `0.93228`
- CatBoost baseline: around `0.97070`
- LightGBM baseline: around `0.97129`
- Expanded-feature LightGBM local result: around `0.97155`

Good file to reference:

- [baseline_results.txt](C:\Users\habib\Desktop\ML%20project\coe546-food-delivery\outputs\baseline_results.txt)

## 7. Validation Strategy

Explain:

- Stratified K-Fold cross-validation was used
- ROC-AUC was the main local metric because it matches the Kaggle competition metric
- Most main experiments used 5-fold validation
- A smaller 3-fold setup was used later for faster tuning experiments

Also include an honest observation:

- Not every local improvement transferred to Kaggle
- This highlighted the importance of using Kaggle submissions carefully

## 8. Kaggle Experiments and Results

Summarize the actual leaderboard progression:

- CatBoost: `0.97136`
- LightGBM: `0.97144`
- 50/50 CatBoost + LightGBM blend: `0.97196`
- 40/60 and 55/45 weighted blends were slightly worse
- Refreshed feature blend scored `0.97184`
- Tuned LightGBM failed badly on Kaggle with `0.91710`
- Target-encoded LightGBM improved to `0.97257`
- Weighted blend with target-encoded LightGBM improved to `0.97260`
- Final top-two target-encoding rank blend reached `0.97261`

Main conclusion:

- The best public result ultimately came from a late-stage rank-blend of the two strongest target-encoding winners

Reference:

- [experiment_log.md](C:\Users\habib\Desktop\ML%20project\coe546-food-delivery\docs\experiment_log.md)

## 9. Final Model

State clearly:

- Final chosen method: final rank-average blend of the two top-performing target-encoding blend winners
- Best public score: `0.97261`
- Best file: [rank_blend_cb_0.50_lgbm_0.50_20260426_042932.csv](C:\Users\habib\Desktop\ML%20project\coe546-food-delivery\submissions\rank_blend_cb_0.50_lgbm_0.50_20260426_042932.csv)

Explain why it was chosen:

- It outperformed each earlier single model
- It outperformed the earlier ensemble family
- It achieved the strongest leaderboard performance among all tested submissions

## 10. Discussion

Talk about what you learned:

- Ensemble methods can outperform individual models in tabular competitions
- Local validation is necessary, but it is not a perfect predictor of leaderboard performance
- Additional feature engineering does not always generalize
- Simpler, reliable methods may outperform more aggressive tuning

## 11. Limitations and Future Work

Possible points:

- More careful time-aware validation could be explored if session chronology matters
- A more systematic ensemble search could be tried
- Better calibration and stack-based blending could be explored
- Additional model families could be tested if time permits

## 12. Conclusion

Close the report with:

- the final best model
- the best Kaggle score
- the overall lesson that combining strong tree-based models gave the best performance
