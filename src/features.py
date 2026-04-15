from __future__ import annotations

import numpy as np
import pandas as pd


TIMESTAMP_COLUMNS = ["f3", "f4", "f5"]
IDENTIFIER_COLUMNS = ["id", "f2"]


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

    if {"active_span_seconds", "session_duration_seconds"}.issubset(frame.columns):
        frame["active_share"] = frame["active_span_seconds"] / frame["session_duration_seconds"].replace(0, np.nan)
        frame["idle_share"] = frame["idle_gap_seconds"] / frame["session_duration_seconds"].replace(0, np.nan)

    if "f3" in frame.columns:
        frame["session_start_hour"] = frame["f3"].dt.hour
        frame["session_start_dayofweek"] = frame["f3"].dt.dayofweek
        frame["session_start_is_weekend"] = frame["session_start_dayofweek"].isin([5, 6]).astype(float)
        frame["session_part_of_day"] = pd.cut(
            frame["session_start_hour"],
            bins=[-1, 5, 11, 16, 21, 24],
            labels=["late_night", "morning", "afternoon", "evening", "night"],
        )

    if {"f10", "f11"}.issubset(frame.columns):
        frame["avg_item_value"] = frame["f11"] / frame["f10"].replace(0, np.nan)
        frame["cart_has_items"] = (frame["f10"] > 0).astype(float)

    if {"f8", "f15"}.issubset(frame.columns):
        frame["declined_offer_ratio"] = frame["f8"] / frame["f15"].replace(0, np.nan)
        frame["offers_remaining_after_declines"] = frame["f15"] - frame["f8"]
        frame["has_any_offer"] = (frame["f15"].fillna(0) > 0).astype(float)

    if {"f11", "f14"}.issubset(frame.columns):
        frame["cart_to_min_order_gap"] = frame["f11"] - frame["f14"]
        frame["meets_min_order"] = (frame["f11"] >= frame["f14"]).astype(float)
        frame["cart_gap_shortfall"] = (frame["f14"] - frame["f11"]).clip(lower=0)
        frame["cart_gap_surplus"] = (frame["f11"] - frame["f14"]).clip(lower=0)

    if {"f13", "f11"}.issubset(frame.columns):
        frame["discount_to_cart_ratio"] = frame["f13"] / frame["f11"].replace(0, np.nan)

    if {"f13", "f14"}.issubset(frame.columns):
        frame["discount_to_min_order_ratio"] = frame["f13"] / frame["f14"].replace(0, np.nan)
        frame["promotion_present"] = frame["f13"].notna().astype(float)

    if {"f13", "f15"}.issubset(frame.columns):
        frame["discount_per_offer_shown"] = frame["f13"] / frame["f15"].replace(0, np.nan)

    if {"f11", "f15"}.issubset(frame.columns):
        frame["cart_value_per_offer"] = frame["f11"] / frame["f15"].replace(0, np.nan)

    if {"f10", "f15"}.issubset(frame.columns):
        frame["items_per_offer"] = frame["f10"] / frame["f15"].replace(0, np.nan)

    if "f17" in frame.columns:
        response = frame["f17"].fillna("missing").astype(str).str.lower()
        frame["promo_response_group"] = response
        frame["promo_response_accepted"] = response.eq("accepted").astype(float)
        frame["promo_response_declined"] = response.eq("declined").astype(float)
        frame["promo_response_ignored"] = response.eq("ignored").astype(float)

    if "f9" in frame.columns:
        customer_type = frame["f9"].fillna("missing").astype(str).str.lower()
        frame["is_returning_customer"] = customer_type.eq("returning").astype(float)

    if {"f12", "f17"}.issubset(frame.columns):
        frame["promo_type_response_combo"] = (
            frame["f12"].fillna("missing").astype(str) + "_" + frame["f17"].fillna("missing").astype(str)
        )

    if {"f7", "f17"}.issubset(frame.columns):
        frame["action_response_combo"] = (
            frame["f7"].fillna("missing").astype(str) + "_" + frame["f17"].fillna("missing").astype(str)
        )

    if {"f7", "f12"}.issubset(frame.columns):
        frame["action_promo_combo"] = (
            frame["f7"].fillna("missing").astype(str) + "_" + frame["f12"].fillna("missing").astype(str)
        )

    for col in ["f12", "f13", "f14", "f15", "f17"]:
        if col in frame.columns:
            frame[f"{col}_missing"] = frame[col].isna().astype(float)

    drop_columns = [col for col in TIMESTAMP_COLUMNS if col in frame.columns]
    drop_columns.extend([col for col in IDENTIFIER_COLUMNS if col in frame.columns])
    frame = frame.drop(columns=drop_columns)
    return frame
