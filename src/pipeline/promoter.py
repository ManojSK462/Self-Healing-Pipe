"""
Model promotion logic.

Decides whether a newly trained candidate should replace the
currently serving model, based on evaluation metrics and safety
guardrails. This is the gatekeeper that prevents regressions
from reaching production.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

from src.serving.model_loader import ModelRegistry
from src.training.evaluator import evaluate_model, compare_models
from src.monitoring.metrics import (
    PROMOTION_EVENTS, update_model_metrics, ACTIVE_MODEL_INFO,
)
import pandas as pd

logger = logging.getLogger(__name__)


class ModelPromoter:
    """
    Handles the promotion decision for candidate models.

    Flow:
    1. Evaluate candidate on the holdout set
    2. Load current model (if any) and evaluate on same holdout
    3. Compare metrics using the evaluator's comparison logic
    4. Promote if candidate wins, log the decision either way
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()

    def attempt_promotion(
        self,
        candidate_model: Any,
        candidate_name: str,
        candidate_params: dict,
        candidate_metrics: Dict[str, float],
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        mlflow_run_id: Optional[str] = None,
    ) -> Tuple[bool, str, Dict]:
        """
        Try to promote the candidate. Returns (promoted, version, verdict).
        """
        # evaluate candidate on holdout
        holdout_metrics = evaluate_model(
            candidate_model, X_holdout, y_holdout, dataset_label="eval"
        )

        # get current model's holdout performance for comparison
        current_metrics = None
        current_version = self.registry.get_active_version()

        if current_version is not None:
            try:
                current_model, _, current_meta = self.registry.load_active()
                current_metrics = evaluate_model(
                    current_model, X_holdout, y_holdout, dataset_label="eval"
                )
                logger.info(
                    f"Current model {current_version}: "
                    f"F1={current_metrics['eval_f1']:.4f}"
                )
            except Exception as e:
                logger.warning(f"Could not load current model for comparison: {e}")
                current_metrics = None

        # run comparison
        verdict = compare_models(holdout_metrics, current_metrics, metric_prefix="eval")

        # always save the model to the registry (we keep history)
        version = self.registry.save_model(
            model=candidate_model,
            metrics={**candidate_metrics, **holdout_metrics},
            model_name=candidate_name,
            params=candidate_params,
            mlflow_run_id=mlflow_run_id,
        )

        if verdict["should_promote"]:
            self.registry.promote(version)
            PROMOTION_EVENTS.inc()
            update_model_metrics(version, holdout_metrics, dataset="eval")

            ACTIVE_MODEL_INFO.info({
                "version": version,
                "model_name": candidate_name,
            })

            logger.info(
                f"PROMOTED {candidate_name} as {version} | "
                f"F1={holdout_metrics['eval_f1']:.4f} | "
                f"Reason: {verdict['reasons']}"
            )
        else:
            logger.info(
                f"REJECTED {candidate_name} ({version}) | "
                f"Reason: {verdict['reasons']}"
            )

        return verdict["should_promote"], version, verdict
