# Notebook Storyline

This notebook should read like a story, not just a sequence of code blocks.

## 1. Title and Objective

Start with:

- what the task is
- what the target variable is
- what the evaluation metric is
- what success means in the Kaggle competition

## 2. Imports and Setup

Include:

- pandas, numpy
- plotting libraries
- sklearn metrics and model selection
- CatBoost and LightGBM

## 3. Load the Data

Show:

- train shape
- test shape
- column names
- target presence in train only

## 4. First Data Audit

Include:

- missing value summary
- target balance
- data types
- brief note on promotion-related missingness

## 5. Exploratory Data Analysis

Recommended subsections:

- target imbalance
- numerical feature summaries
- category-level target rates
- time-related behavior

Add a short interpretation after each chart or table.

## 6. Feature Engineering

Introduce this section clearly:

- state that you are creating derived features from existing columns
- explain why raw timestamps and promotion data benefit from transformation

Show a few representative examples:

- session duration
- average item value
- cart gap to minimum order
- promotion response indicators

## 7. Baseline Modeling

Structure:

- logistic regression baseline
- CatBoost baseline
- LightGBM baseline

For each:

- briefly explain why it was tested
- show mean ROC-AUC
- keep code clean and reusable

## 8. Model Comparison

Present a summary table comparing:

- model name
- validation setup
- mean ROC-AUC
- notes

This section should make it obvious why CatBoost and LightGBM became the finalists.

## 9. Kaggle-Oriented Experimentation

Briefly document:

- CatBoost public score
- LightGBM public score
- 50/50 blend public score
- weighted blend attempts
- failed feature-refresh and tuned-LightGBM attempts

This section is important because it shows real decision-making, not just local metrics.

## 10. Final Model Choice

State clearly:

- The final chosen method was the 50/50 CatBoost + LightGBM blend
- It achieved the best public leaderboard score of `0.97196`
- It outperformed both single models and later riskier variations

## 11. Submission Generation

Show:

- how the final submission file is created
- that predictions are probabilities
- that the final output format is `id, order_placed`

## 12. Conclusion

Wrap up with:

- what worked best
- what did not generalize
- what you learned from the competition

## Notebook Writing Advice

- Keep code sections short and grouped by purpose
- Add markdown between major code blocks
- After every major result, write 2 to 4 lines interpreting it
- Do not hide failed experiments completely; summarize the ones that matter
- End with the actual final chosen pipeline, not just the latest experiment
