# Starter Notebook Outline

## 1. Problem Statement

- Predict `order_placed` for each session
- Competition metric: ROC-AUC
- Submission format: `id, order_placed`

## 2. Imports and Configuration

- pandas, numpy
- matplotlib, seaborn
- sklearn metrics and model selection
- chosen gradient boosting libraries

## 3. Load Data

- Read training data
- Read test data
- Read sample submission
- Inspect shapes and column names

## 4. Initial Data Audit

- Missing values
- Data types
- Duplicate rows
- Target balance

## 5. Exploratory Data Analysis

- Numerical summaries
- Categorical summaries
- Target rate by category
- Timestamp inspection
- Correlation or feature importance previews

## 6. Preprocessing

- Timestamp parsing
- Missing value handling
- Encoding
- Column selection

## 7. Feature Engineering

- Session duration
- Time-of-day and day-of-week
- Cart and promotion ratios
- Qualification and response indicators

## 8. Validation Strategy

- Stratified K-Fold
- ROC-AUC scoring
- Leakage prevention notes

## 9. Baseline Models

- Logistic Regression baseline
- Tree-based model baseline
- Gradient boosting baseline

## 10. Model Comparison

- Fold scores
- Mean AUC
- Error analysis

## 11. Final Model and Submission

- Train chosen model
- Predict probabilities on test data
- Save Kaggle submission CSV

## 12. Conclusions

- Best findings
- Main limitations
- Next experiments
