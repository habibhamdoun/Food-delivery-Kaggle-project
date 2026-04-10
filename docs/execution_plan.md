# Execution Plan

## Deadlines

- Kaggle competition closes: April 22, 2026 at 9:00 AM
- Report, notebook, and presentation due: April 27, 2026 at 9:00 AM

## Phase 1: Project Setup

Goals:
- Create a clean workspace for the project
- Download all Kaggle files
- Confirm train and test schema

Checklist:
- Create folder structure
- Add starter notebook
- Add experiment log
- Verify required columns and submission format

## Phase 2: EDA

Questions to answer:
- How imbalanced is `order_placed`?
- Which columns have missing values?
- How informative are promotion response, cart value, and customer type?
- What time-based trends exist in the session timestamps?

Outputs:
- Missing-value table
- Target distribution chart
- Numeric distribution plots
- Category-level conversion summaries

## Phase 3: Baseline Pipeline

Goals:
- Convert timestamps safely
- Build fold-based validation with ROC-AUC
- Train the first baseline models

Initial models:
- Logistic Regression
- Random Forest or ExtraTrees
- CatBoost
- LightGBM or XGBoost if available

## Phase 4: Feature Engineering

Priority features:
- Session duration
- Time since last activity
- Hour of day
- Day of week
- Cart value per item
- Promotion decline ratio
- Gap between cart value and minimum qualifying spend
- Promotion qualification flag

Potential advanced features:
- Leakage-safe customer aggregates by `id`
- Interaction features between response, promotion type, and customer type
- Count and intensity ratios using promotion and cart fields

## Phase 5: Optimization

Goals:
- Tune only the strongest models
- Compare single-model and simple ensemble performance
- Use Kaggle submissions sparingly and intentionally

Rules:
- Do not submit binary predictions
- Do not trust public leaderboard shifts blindly
- Log every submission with local CV and public score

## Phase 6: Final Deliverables

Notebook:
- Keep it clean and runnable
- Explain decisions, not just code

Report:
- Problem understanding
- EDA
- Preprocessing
- Feature engineering
- Modeling and validation
- Challenges and lessons learned

Presentation:
- Problem
- Data and insights
- Model pipeline
- Results
- Takeaways
