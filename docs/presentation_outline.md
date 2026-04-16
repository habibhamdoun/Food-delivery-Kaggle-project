# Presentation Outline

## Slide 1: Title

- Project title
- Team members
- Course name

## Slide 2: Problem Statement

- Predict whether a user places an order during a session
- Use session behavior and promotion interaction data
- Competition metric: ROC-AUC

## Slide 3: Dataset Overview

- Train size: `297,236`
- Test size: `99,639`
- Target: `order_placed`
- Data includes timing, cart, promotion, and user-context features

## Slide 4: Main Challenges

- Strong class imbalance
- Mixed numeric and categorical variables
- Missing values in promotion-related features
- Need to generalize to hidden Kaggle test data

## Slide 5: EDA Highlights

- Target rate is about `2.91%`
- Promotion-related fields have significant missing values
- Cart value and qualification-related features appear important
- Session duration is strongly related to order completion

## Slide 6: Feature Engineering

Examples:

- Session duration
- Idle gap
- Average item value
- Gap to minimum order
- Promotion response indicators
- Offer-related ratios

Short message:

- We transformed raw session and promotion data into more meaningful behavioral signals

## Slide 7: Models Tried

- Logistic Regression
- CatBoost
- LightGBM
- CatBoost + LightGBM ensemble

## Slide 8: Local Validation Results

Show a compact table:

- Logistic: around `0.932`
- CatBoost: around `0.971`
- LightGBM: around `0.971`
- Ensemble: best Kaggle result later came from blending

## Slide 9: Kaggle Submission Results

Use a small timeline:

- CatBoost: `0.97136`
- LightGBM: `0.97144`
- 50/50 blend: `0.97196`
- Other weighted blends: slightly worse
- Feature-refresh blend: worse on Kaggle
- Tuned LightGBM: failed badly on Kaggle

## Slide 10: Final Model

- Final choice: 50/50 CatBoost + LightGBM blend
- Best public score: `0.97196`
- Why chosen:
  - best leaderboard result
  - stronger than individual models
  - more reliable than later risky experiments

## Slide 11: What We Learned

- Ensemble learning was highly effective
- Strong local validation does not guarantee better Kaggle performance
- More complex tuning does not always improve generalization
- Careful experiment tracking was important

## Slide 12: Conclusion

- Restate the final result
- Mention best score and final method
- End with a short reflection on business relevance and next steps

## Presentation Tips

- Keep slides visual and not text-heavy
- Use only the strongest 1 to 2 charts from EDA
- Focus on decision-making, not every technical detail
- Explain failures briefly because they strengthen your credibility
