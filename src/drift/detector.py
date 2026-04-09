"""
Population Stability Index (PSI) based data drift detection.

PSI measures how much the distribution of a variable has shifted
between two datasets. It's the standard metric for detecting
covariate drift in production ML systems.

PSI interpretation:
  < 0.10  — no significant shift
  0.10–0.25 — moderate shift, monitor closely
  > 0.25  — significant shift, likely need to retrain

We compute PSI per-feature and also track an aggregate score.
When aggregate PSI crosses the retrain threshold, the orchestrator
kicks off an automated retraining cycle.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from config.settings import (
    PSI_THRESHOLD, PSI_RETRAIN_THRESHOLD, PSI_NUM_BINS, FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Container for a drift detection run."""
    timestamp: str
    feature_psi: Dict[str, float]
    aggregate_psi: float
    drifted_features: List[str]
    alert_level: str  # "none", "warning", "critical"
    should_retrain: bool
    reference_stats: Dict = field(default_factory=dict)
    current_stats: Dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Drift Report ({self.timestamp})",
            f"  Aggregate PSI: {self.aggregate_psi:.4f} [{self.alert_level}]",
            f"  Retrain recommended: {self.should_retrain}",
        ]
        if self.drifted_features:
            lines.append(f"  Drifted features ({len(self.drifted_features)}):")
            for feat in self.drifted_features:
                lines.append(f"    - {feat}: PSI={self.feature_psi[feat]:.4f}")
        return "\n".join(lines)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = PSI_NUM_BINS,
) -> float:
    """
    Compute PSI between reference and current distributions.

    Uses quantile-based binning on the reference distribution so
    each bin has roughly equal representation. This avoids the
    problem with equal-width bins where empty bins cause infinity.
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) < n_bins or len(current) < n_bins:
        logger.warning("Too few samples for reliable PSI — returning 0")
        return 0.0

    # quantile-based bin edges from the reference
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges = np.unique(bin_edges)  # collapse duplicates

    if len(bin_edges) < 3:
        # degenerate case — nearly constant feature
        return 0.0

    ref_counts = np.histogram(reference, bins=bin_edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=bin_edges)[0].astype(float)

    # convert to proportions with smoothing to avoid log(0)
    eps = 1e-6
    ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))
    cur_pct = (cur_counts + eps) / (cur_counts.sum() + eps * len(cur_counts))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


class DriftDetector:
    """
    Monitors feature distributions for drift against a reference dataset.

    The reference is typically the training data distribution. Each time
    new production data comes in, we compute PSI per-feature and decide
    if retraining is warranted.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        psi_warn_threshold: float = PSI_THRESHOLD,
        psi_retrain_threshold: float = PSI_RETRAIN_THRESHOLD,
    ):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.psi_warn = psi_warn_threshold
        self.psi_retrain = psi_retrain_threshold

        # store reference distributions
        self._reference = {}
        for col in self.feature_columns:
            if col in reference_data.columns:
                self._reference[col] = reference_data[col].values.astype(float)
            else:
                logger.warning(f"Reference data missing column: {col}")

        self._history: List[DriftReport] = []

    def check(self, current_data: pd.DataFrame) -> DriftReport:
        """
        Run drift detection against current data batch.
        """
        from datetime import datetime

        feature_psi = {}
        drifted = []

        for col in self.feature_columns:
            if col not in self._reference:
                continue
            if col not in current_data.columns:
                logger.warning(f"Current data missing column: {col}")
                continue

            current_vals = current_data[col].values.astype(float)
            psi = compute_psi(self._reference[col], current_vals)
            feature_psi[col] = psi

            if psi >= self.psi_warn:
                drifted.append(col)

        agg_psi = float(np.mean(list(feature_psi.values()))) if feature_psi else 0.0

        if agg_psi >= self.psi_retrain:
            alert_level = "critical"
        elif agg_psi >= self.psi_warn:
            alert_level = "warning"
        else:
            alert_level = "none"

        should_retrain = agg_psi >= self.psi_retrain

        report = DriftReport(
            timestamp=datetime.utcnow().isoformat(),
            feature_psi=feature_psi,
            aggregate_psi=agg_psi,
            drifted_features=drifted,
            alert_level=alert_level,
            should_retrain=should_retrain,
        )

        self._history.append(report)

        if should_retrain:
            logger.warning(f"DRIFT CRITICAL — aggregate PSI={agg_psi:.4f}, triggering retrain")
        elif alert_level == "warning":
            logger.warning(f"DRIFT WARNING — aggregate PSI={agg_psi:.4f}")
        else:
            logger.info(f"Drift check OK — aggregate PSI={agg_psi:.4f}")

        return report

    @property
    def history(self) -> List[DriftReport]:
        return self._history

    def feature_psi_summary(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame ranking features by PSI for quick triage."""
        rows = []
        for col in self.feature_columns:
            if col not in self._reference or col not in current_data.columns:
                continue
            psi = compute_psi(self._reference[col], current_data[col].values.astype(float))
            ref_mean = float(np.mean(self._reference[col]))
            cur_mean = float(current_data[col].mean())
            rows.append({
                "feature": col,
                "psi": psi,
                "ref_mean": ref_mean,
                "cur_mean": cur_mean,
                "mean_shift_pct": abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-8) * 100,
                "drifted": psi >= self.psi_warn,
            })
        return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
