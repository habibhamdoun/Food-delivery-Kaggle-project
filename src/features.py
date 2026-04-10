from __future__ import annotations

import numpy as np
import pandas as pd


TIMESTAMP_COLUMNS = ["f3", "f4", "f5"]


def _coerce_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for col in TIMESTAMP_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=False)
    return frame


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = _coerce_datetime_columns(df)

    if {"f3", "f4"}.issubset(frame.columns):
        duration = (frame["f4"] - frame["f3"]).dt.total_seconds()
        frame["session_duration_seconds"] = duration

    if {"f3", "f5"}.issubset(frame.columns):
        active_span = (frame["f5"] - frame["f3"]).dt.total_seconds()
        frame["active_span_seconds"] = active_span

    if {"f4", "f5"}.issubset(frame.columns):
        idle_gap = (frame["f4"] - frame["f5"]).dt.total_seconds()
        frame["idle_gap_seconds"] = idle_gap

    if "f3" in frame.columns:
        frame["session_start_hour"] = frame["f3"].dt.hour
        frame["session_start_dayofweek"] = frame["f3"].dt.dayofweek
        frame["session_start_is_weekend"] = frame["session_start_dayofweek"].isin([5, 6]).astype(float)

    if {"f10", "f11"}.issubset(frame.columns):
        frame["avg_item_value"] = frame["f11"] / frame["f10"].replace(0, np.nan)

    if {"f8", "f15"}.issubset(frame.columns):
        frame["declined_offer_ratio"] = frame["f8"] / frame["f15"].replace(0, np.nan)

    if {"f11", "f14"}.issubset(frame.columns):
        frame["cart_to_min_order_gap"] = frame["f11"] - frame["f14"]
        frame["meets_min_order"] = (frame["f11"] >= frame["f14"]).astype(float)

    if {"f13", "f11"}.issubset(frame.columns):
        frame["discount_to_cart_ratio"] = frame["f13"] / frame["f11"].replace(0, np.nan)

    if {"f13", "f14"}.issubset(frame.columns):
        frame["discount_to_min_order_ratio"] = frame["f13"] / frame["f14"].replace(0, np.nan)

    drop_columns = [col for col in TIMESTAMP_COLUMNS if col in frame.columns]
    frame = frame.drop(columns=drop_columns)
    return frame
