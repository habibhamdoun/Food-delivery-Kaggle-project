from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
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


def _prepare_tree_frame(
    X: pd.DataFrame,
    reference_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    frame = X.copy()

    for col in frame.columns:
        if pd.api.types.is_bool_dtype(frame[col]):
            frame[col] = frame[col].astype(float)

    categorical_columns = frame.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_columns = [col for col in frame.columns if col not in categorical_columns]

    for col in categorical_columns:
        frame[col] = frame[col].fillna("missing").astype("category")
        if reference_frame is not None:
            categories = reference_frame[col].cat.categories
            frame[col] = pd.Categorical(frame[col], categories=categories)

    for col in numeric_columns:
        if reference_frame is None:
            fill_value = frame[col].median()
        else:
            fill_value = reference_frame[col].median()
        frame[col] = frame[col].fillna(fill_value)

    return frame, categorical_columns


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


def prepare_training_data(
    df: pd.DataFrame,
    feature_builder=build_features,
) -> tuple[pd.DataFrame, pd.Series]:
    frame = feature_builder(df)
    y = frame[TARGET_COLUMN].astype(int)
    feature_columns = [col for col in frame.columns if col != TARGET_COLUMN]
    X = frame[feature_columns].copy()

    if ID_COLUMN in X.columns:
        X[ID_COLUMN] = X[ID_COLUMN].astype(str)

    return X, y


def prepare_inference_data(
    df: pd.DataFrame,
    feature_builder=build_features,
) -> pd.DataFrame:
    frame = feature_builder(df)
    if TARGET_COLUMN in frame.columns:
        frame = frame.drop(columns=[TARGET_COLUMN])

    if ID_COLUMN in frame.columns:
        frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)

    return frame


def evaluate_logistic_baseline(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    feature_builder=build_features,
) -> BaselineResult:
    X, y = prepare_training_data(train_df, feature_builder=feature_builder)
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
    model_params: dict | None = None,
    feature_builder=build_features,
) -> BaselineResult:
    X, y = prepare_training_data(train_df, feature_builder=feature_builder)
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

        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": 400,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": random_state,
            "verbose": False,
        }
        if model_params:
            params.update(model_params)

        model = CatBoostClassifier(
            **params,
        )

        model.fit(
            X_train,
            y_train,
            cat_features=categorical_indices,
        )
        valid_proba = model.predict_proba(X_valid)[:, 1]
        fold_scores.append(roc_auc_score(y_valid, valid_proba))

    return BaselineResult(fold_scores=fold_scores)


def evaluate_lightgbm_baseline(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    model_params: dict | None = None,
    feature_builder=build_features,
) -> BaselineResult:
    X, y = prepare_training_data(train_df, feature_builder=feature_builder)
    X, categorical_columns = _prepare_tree_frame(X)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: list[float] = []

    for train_idx, valid_idx in cv.split(X, y):
        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        X_train, categorical_columns = _prepare_tree_frame(X_train)
        X_valid, _ = _prepare_tree_frame(X_valid, reference_frame=X_train)

        params = {
            "objective": "binary",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "verbose": -1,
        }
        if model_params:
            params.update(model_params)

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, categorical_feature=categorical_columns)
        valid_proba = model.predict_proba(X_valid)[:, 1]
        fold_scores.append(roc_auc_score(y_valid, valid_proba))

    return BaselineResult(fold_scores=fold_scores)


def fit_catboost_model(
    train_df: pd.DataFrame,
    random_state: int = 42,
    model_params: dict | None = None,
    feature_builder=build_features,
) -> tuple[CatBoostClassifier, pd.DataFrame, list[str]]:
    X, y = prepare_training_data(train_df, feature_builder=feature_builder)
    X = _normalize_for_catboost(X)

    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()
    categorical_indices = [X.columns.get_loc(col) for col in categorical_columns]

    for col in categorical_columns:
        X[col] = X[col].fillna("missing").astype(str)

    numeric_columns = [col for col in X.columns if col not in categorical_columns]
    for col in numeric_columns:
        X[col] = X[col].fillna(X[col].median())

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 400,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": random_state,
        "verbose": False,
    }
    if model_params:
        params.update(model_params)

    model = CatBoostClassifier(**params)
    model.fit(X, y, cat_features=categorical_indices)
    return model, X, categorical_columns


def predict_with_catboost(
    model: CatBoostClassifier,
    test_df: pd.DataFrame,
    train_feature_frame: pd.DataFrame,
    categorical_columns: list[str],
    feature_builder=build_features,
) -> np.ndarray:
    X_test = prepare_inference_data(test_df, feature_builder=feature_builder)
    X_test = _normalize_for_catboost(X_test)
    X_test = X_test.reindex(columns=train_feature_frame.columns)

    for col in categorical_columns:
        X_test[col] = X_test[col].fillna("missing").astype(str)

    numeric_columns = [col for col in X_test.columns if col not in categorical_columns]
    for col in numeric_columns:
        median = train_feature_frame[col].median()
        X_test[col] = X_test[col].fillna(median)

    return model.predict_proba(X_test)[:, 1]


def fit_lightgbm_model(
    train_df: pd.DataFrame,
    random_state: int = 42,
    model_params: dict | None = None,
    feature_builder=build_features,
) -> tuple[LGBMClassifier, pd.DataFrame, list[str]]:
    X, y = prepare_training_data(train_df, feature_builder=feature_builder)
    X, categorical_columns = _prepare_tree_frame(X)

    params = {
        "objective": "binary",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "verbose": -1,
    }
    if model_params:
        params.update(model_params)

    model = LGBMClassifier(**params)
    model.fit(X, y, categorical_feature=categorical_columns)
    return model, X, categorical_columns


def predict_with_lightgbm(
    model: LGBMClassifier,
    test_df: pd.DataFrame,
    train_feature_frame: pd.DataFrame,
    categorical_columns: list[str],
    feature_builder=build_features,
) -> np.ndarray:
    X_test = prepare_inference_data(test_df, feature_builder=feature_builder)
    X_test = X_test.reindex(columns=train_feature_frame.columns)
    X_test, _ = _prepare_tree_frame(X_test, reference_frame=train_feature_frame)
    return model.predict_proba(X_test)[:, 1]
