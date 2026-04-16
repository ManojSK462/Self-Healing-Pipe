"""
Model evaluation and comparison.

Handles side-by-side evaluation of candidate models against the
currently deployed model, with promotion decisions based on
precision-recall tradeoffs and minimum improvement thresholds.
"""

import logging
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from typing import Any, Dict, Optional, Tuple

from config.settings import MIN_PRECISION, MIN_RECALL, MIN_F1_IMPROVEMENT

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_label: str = "eval",
) -> Dict[str, float]:
    """Full evaluation of a single model on a dataset."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        f"{dataset_label}_precision": precision_score(y, y_pred, zero_division=0),
        f"{dataset_label}_recall": recall_score(y, y_pred, zero_division=0),
        f"{dataset_label}_f1": f1_score(y, y_pred, zero_division=0),
        f"{dataset_label}_roc_auc": roc_auc_score(y, y_prob),
        f"{dataset_label}_avg_precision": average_precision_score(y, y_prob),
    }

    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics[f"{dataset_label}_true_positives"] = int(tp)
        metrics[f"{dataset_label}_false_positives"] = int(fp)
        metrics[f"{dataset_label}_true_negatives"] = int(tn)
        metrics[f"{dataset_label}_false_negatives"] = int(fn)
        # false positive rate matters a lot in fraud — too many false alarms
        # and the ops team stops trusting the system
        metrics[f"{dataset_label}_fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return metrics


def compare_models(
    candidate_metrics: Dict[str, float],
    current_metrics: Optional[Dict[str, float]],
    metric_prefix: str = "eval",
) -> Dict[str, Any]:
    """
    Compare candidate model against currently deployed model.
    Returns a verdict dict with the decision and reasoning.
    """
    verdict = {
        "should_promote": False,
        "reasons": [],
        "candidate_metrics": candidate_metrics,
        "current_metrics": current_metrics,
    }

    cand_precision = candidate_metrics.get(f"{metric_prefix}_precision", 0)
    cand_recall = candidate_metrics.get(f"{metric_prefix}_recall", 0)
    cand_f1 = candidate_metrics.get(f"{metric_prefix}_f1", 0)

    # gate: minimum performance bar
    if cand_precision < MIN_PRECISION:
        verdict["reasons"].append(
            f"Precision {cand_precision:.4f} below minimum {MIN_PRECISION}"
        )
        return verdict

    if cand_recall < MIN_RECALL:
        verdict["reasons"].append(
            f"Recall {cand_recall:.4f} below minimum {MIN_RECALL}"
        )
        return verdict

    # if no current model, any model passing the gates gets promoted
    if current_metrics is None:
        verdict["should_promote"] = True
        verdict["reasons"].append("No existing model — promoting first candidate that passes gates.")
        return verdict

    current_f1 = current_metrics.get(f"{metric_prefix}_f1", 0)
    f1_delta = cand_f1 - current_f1

    if f1_delta < MIN_F1_IMPROVEMENT:
        verdict["reasons"].append(
            f"F1 improvement {f1_delta:.4f} below threshold {MIN_F1_IMPROVEMENT}"
        )
        return verdict

    # check we're not trading too much precision for recall or vice versa
    current_precision = current_metrics.get(f"{metric_prefix}_precision", 0)
    current_recall = current_metrics.get(f"{metric_prefix}_recall", 0)

    precision_drop = current_precision - cand_precision
    recall_drop = current_recall - cand_recall

    if precision_drop > 0.05:
        verdict["reasons"].append(
            f"Precision dropped by {precision_drop:.4f} — too much regression"
        )
        return verdict

    if recall_drop > 0.08:
        verdict["reasons"].append(
            f"Recall dropped by {recall_drop:.4f} — too much regression"
        )
        return verdict

    verdict["should_promote"] = True
    verdict["reasons"].append(
        f"F1 improved by {f1_delta:.4f} without unacceptable precision/recall regression."
    )
    return verdict


def generate_evaluation_report(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "candidate",
) -> str:
    """Human-readable evaluation report."""
    y_pred = model.predict(X)

    report_lines = [
        f"=== Evaluation Report: {model_name} ===",
        f"Samples: {len(y)} (fraud rate: {y.mean():.3f})",
        "",
        classification_report(y, y_pred, target_names=["legitimate", "fraud"]),
        "",
        "Confusion Matrix:",
    ]

    cm = confusion_matrix(y, y_pred)
    report_lines.append(f"  TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    report_lines.append(f"  FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")

    return "\n".join(report_lines)
