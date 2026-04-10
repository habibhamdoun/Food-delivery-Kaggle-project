from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ID_COLUMN, TARGET_COLUMN
from .features import build_features


@dataclass
class BaselineResult:
    fold_scores: list[float]

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.fold_scores))


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = build_features(df)
    y = frame[TARGET_COLUMN].astype(int)
    feature_columns = [col for col in frame.columns if col != TARGET_COLUMN]
    X = frame[feature_columns].copy()

    if ID_COLUMN in X.columns:
        X[ID_COLUMN] = X[ID_COLUMN].astype(str)

    return X, y


def evaluate_logistic_baseline(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> BaselineResult:
    X, y = prepare_training_data(train_df)
    preprocessor = build_preprocessor(X)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: list[float] = []

    for train_idx, valid_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model.fit(X_train, y_train)
        valid_proba = model.predict_proba(X_valid)[:, 1]
        fold_scores.append(roc_auc_score(y_valid, valid_proba))

    return BaselineResult(fold_scores=fold_scores)
