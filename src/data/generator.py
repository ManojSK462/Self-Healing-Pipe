"""
Synthetic credit-card transaction generator.

Produces a realistic-ish fraud-detection dataset with controllable drift.
The drift knobs let us simulate distribution shift over time so the
self-healing pipeline has something real to react to.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


_MERCHANT_CATEGORIES = {
    0: "grocery",
    1: "gas_station",
    2: "restaurant",
    3: "online_retail",
    4: "travel",
    5: "entertainment",
    6: "healthcare",
    7: "utilities",
}

_NUM_CATEGORIES = len(_MERCHANT_CATEGORIES)


def _base_transaction_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the core transaction feature matrix."""
    tx_amount = np.exp(rng.normal(3.8, 1.1, n))  # log-normal spend
    tx_amount = np.clip(tx_amount, 0.50, 15000.0)

    merchant_category = rng.integers(0, _NUM_CATEGORIES, n)

    hours = rng.normal(14, 4, n)  # peak around 2pm
    tx_hour = np.clip(hours, 0, 23).astype(int)
    tx_day_of_week = rng.integers(0, 7, n)

    distance_from_home = np.abs(rng.exponential(8.0, n))
    distance_from_last_tx = np.abs(rng.exponential(3.5, n))

    # ratio to median purchase — centered around 1 for legit, higher for fraud (set later)
    ratio_to_median_price = np.abs(rng.normal(1.0, 0.4, n))

    is_chip_used = rng.binomial(1, 0.72, n)
    is_pin_used = rng.binomial(1, 0.45, n)
    is_online = rng.binomial(1, 0.35, n)

    # engineered aggregates (simulated)
    tx_frequency_1h = rng.poisson(1.2, n).astype(float)
    tx_amount_avg_7d = tx_amount * rng.uniform(0.6, 1.4, n)
    tx_amount_std_7d = tx_amount * rng.uniform(0.1, 0.6, n)

    return pd.DataFrame({
        "tx_amount": tx_amount,
        "tx_hour": tx_hour,
        "tx_day_of_week": tx_day_of_week,
        "merchant_category": merchant_category,
        "distance_from_home": distance_from_home,
        "distance_from_last_tx": distance_from_last_tx,
        "ratio_to_median_price": ratio_to_median_price,
        "is_chip_used": is_chip_used,
        "is_pin_used": is_pin_used,
        "is_online": is_online,
        "tx_frequency_1h": tx_frequency_1h,
        "tx_amount_avg_7d": tx_amount_avg_7d,
        "tx_amount_std_7d": tx_amount_std_7d,
    })


def _assign_fraud_labels(df: pd.DataFrame, fraud_ratio: float,
                         rng: np.random.Generator) -> pd.Series:
    """
    Assign fraud labels using a probability model rather than random sampling.
    Higher amounts, online, no pin, unusual hours → more likely fraud.
    """
    score = np.zeros(len(df))

    # large transactions relative to median
    score += np.where(df["ratio_to_median_price"] > 2.0, 0.3, 0.0)
    score += np.where(df["tx_amount"] > 500, 0.15, 0.0)

    # online with no pin is riskier
    score += np.where((df["is_online"] == 1) & (df["is_pin_used"] == 0), 0.2, 0.0)

    # unusual hours (late night)
    score += np.where((df["tx_hour"] < 5) | (df["tx_hour"] > 22), 0.15, 0.0)

    # far from home
    score += np.where(df["distance_from_home"] > 30, 0.1, 0.0)

    # high tx frequency burst
    score += np.where(df["tx_frequency_1h"] > 3, 0.1, 0.0)

    # convert scores to probabilities, calibrated to target fraud_ratio
    probs = 1 / (1 + np.exp(-(score - 0.3) * 5))
    # scale to hit target ratio approximately
    probs = probs * (fraud_ratio / probs.mean()) if probs.mean() > 0 else probs
    probs = np.clip(probs, 0.001, 0.95)

    labels = rng.binomial(1, probs)
    return pd.Series(labels, name="is_fraud")


def _inject_fraud_patterns(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Make fraud rows look more fraudulent — amplify the signal."""
    df = df.copy()
    fraud_mask = df["is_fraud"] == 1

    n_fraud = fraud_mask.sum()
    if n_fraud == 0:
        return df

    df.loc[fraud_mask, "tx_amount"] *= rng.uniform(1.5, 4.0, n_fraud)
    df.loc[fraud_mask, "distance_from_home"] *= rng.uniform(1.3, 3.0, n_fraud)
    df.loc[fraud_mask, "ratio_to_median_price"] *= rng.uniform(1.5, 3.5, n_fraud)
    df.loc[fraud_mask, "tx_frequency_1h"] += rng.poisson(2, n_fraud)

    # fraud transactions less likely to use chip/pin
    df.loc[fraud_mask, "is_chip_used"] = rng.binomial(1, 0.25, n_fraud)
    df.loc[fraud_mask, "is_pin_used"] = rng.binomial(1, 0.10, n_fraud)

    return df


def _add_timestamps(df: pd.DataFrame, start_date: datetime,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Assign realistic timestamps spread across the date range."""
    n = len(df)
    offsets = np.sort(rng.uniform(0, 30 * 24 * 3600, n))  # ~30 days
    timestamps = [start_date + timedelta(seconds=float(s)) for s in offsets]
    df = df.copy()
    df["event_timestamp"] = timestamps
    df["entity_id"] = [f"user_{i % 800:04d}" for i in range(n)]  # ~800 users
    return df


def generate_training_data(
    n_samples: int = 15000,
    fraud_ratio: float = 0.035,
    seed: int = 42,
    start_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Generate the baseline training dataset."""
    rng = np.random.default_rng(seed)
    if start_date is None:
        start_date = datetime(2024, 1, 1)

    df = _base_transaction_features(n_samples, rng)
    df["is_fraud"] = _assign_fraud_labels(df, fraud_ratio, rng)
    df = _inject_fraud_patterns(df, rng)
    df = _add_timestamps(df, start_date, rng)
    df = df.reset_index(drop=True)

    return df


def generate_drifted_data(
    n_samples: int = 3000,
    fraud_ratio: float = 0.035,
    seed: int = 99,
    drift_intensity: float = 1.0,
    start_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generate data with controlled distribution shift.

    drift_intensity controls how much the distributions deviate:
      0.0 = identical to training
      1.0 = moderate drift (should trigger PSI alerts)
      2.0+ = severe drift (should trigger retraining)
    """
    rng = np.random.default_rng(seed)
    if start_date is None:
        start_date = datetime(2024, 3, 1)

    df = _base_transaction_features(n_samples, rng)

    # apply drift: shift the distributions
    amount_shift = 1.0 + 0.4 * drift_intensity
    df["tx_amount"] *= amount_shift

    # merchants shift — more online retail
    online_boost = int(n_samples * 0.15 * drift_intensity)
    if online_boost > 0:
        boost_idx = rng.choice(n_samples, min(online_boost, n_samples), replace=False)
        df.loc[boost_idx, "merchant_category"] = 3  # online_retail
        df.loc[boost_idx, "is_online"] = 1

    # transaction time shifts later
    df["tx_hour"] = np.clip(df["tx_hour"] + int(2 * drift_intensity), 0, 23).astype(int)

    # distance patterns change
    df["distance_from_home"] *= (1.0 + 0.3 * drift_intensity)

    # frequency increases
    df["tx_frequency_1h"] += drift_intensity * 0.8

    # ratio shifts
    df["ratio_to_median_price"] *= (1.0 + 0.2 * drift_intensity)

    # slightly different fraud ratio under drift
    drifted_fraud_ratio = fraud_ratio * (1 + 0.5 * drift_intensity)
    df["is_fraud"] = _assign_fraud_labels(df, drifted_fraud_ratio, rng)
    df = _inject_fraud_patterns(df, rng)
    df = _add_timestamps(df, start_date, rng)

    return df.reset_index(drop=True)
