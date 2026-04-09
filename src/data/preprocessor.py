"""
Data preprocessing and validation.

Handles cleaning, type enforcement, and train/test splitting with
stratification to preserve the fraud class ratio.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

from config.settings import FEATURE_COLUMNS, TARGET_COLUMN, TEST_SPLIT_RATIO, RANDOM_SEED

logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Check that required columns exist and coerce types."""
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    # force numeric types on feature columns
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    na_counts = df[FEATURE_COLUMNS].isna().sum()
    cols_with_na = na_counts[na_counts > 0]
    if len(cols_with_na) > 0:
        logger.warning(f"NaN values found after coercion: {cols_with_na.to_dict()}")
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def compute_feature_stats(df: pd.DataFrame) -> dict:
    """Compute summary stats for monitoring and drift comparison."""
    stats = {}
    for col in FEATURE_COLUMNS:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "p25": float(df[col].quantile(0.25)),
            "p50": float(df[col].quantile(0.50)),
            "p75": float(df[col].quantile(0.75)),
        }
    stats["_target_rate"] = float(df[TARGET_COLUMN].mean())
    stats["_n_samples"] = len(df)
    return stats


def split_data(
    df: pd.DataFrame,
    test_ratio: float = TEST_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split preserving fraud ratio."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=df[TARGET_COLUMN],
    )
    logger.info(
        f"Split: train={len(train_df)} (fraud={train_df[TARGET_COLUMN].mean():.3f}), "
        f"test={len(test_df)} (fraud={test_df[TARGET_COLUMN].mean():.3f})"
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix and target vector."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
