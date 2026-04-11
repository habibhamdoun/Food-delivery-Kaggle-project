from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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


def _normalize_for_catboost(X: pd.DataFrame) -> pd.DataFrame:
    frame = X.copy()
    for col in frame.columns:
        if pd.api.types.is_bool_dtype(frame[col]):
            frame[col] = frame[col].astype(float)
    return frame


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


def evaluate_catboost_baseline(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> BaselineResult:
    X, y = prepare_training_data(train_df)
    X = _normalize_for_catboost(X)

    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()
    categorical_indices = [X.columns.get_loc(col) for col in categorical_columns]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: list[float] = []

    for train_idx, valid_idx in cv.split(X, y):
        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        for col in categorical_columns:
            X_train[col] = X_train[col].fillna("missing").astype(str)
            X_valid[col] = X_valid[col].fillna("missing").astype(str)

        numeric_columns = [col for col in X.columns if col not in categorical_columns]
        for col in numeric_columns:
            median = X_train[col].median()
            X_train[col] = X_train[col].fillna(median)
            X_valid[col] = X_valid[col].fillna(median)

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=400,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=False,
        )

        model.fit(
            X_train,
            y_train,
            cat_features=categorical_indices,
        )
        valid_proba = model.predict_proba(X_valid)[:, 1]
        fold_scores.append(roc_auc_score(y_valid, valid_proba))

    return BaselineResult(fold_scores=fold_scores)
