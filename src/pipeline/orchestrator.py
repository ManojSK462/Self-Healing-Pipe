"""
Self-healing pipeline orchestrator.

This is the main control loop. It:
1. Generates/loads training data
2. Sets up the feature store
3. Runs the initial training sweep
4. Promotes the best model
5. Starts serving
6. Continuously monitors for drift
7. When drift is detected, triggers retraining and re-promotion

The "self-healing" part: the pipeline detects when its own predictions
are degrading due to data drift and autonomously retrains + redeploys
without human intervention.
"""

import logging
import time
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from config.settings import (
    DATA_DIR, ARTIFACTS_DIR, FEATURE_COLUMNS, TARGET_COLUMN,
    NUM_TRAIN_SAMPLES, NUM_DRIFT_SAMPLES, FRAUD_RATIO, RANDOM_SEED,
)
from src.data.generator import generate_training_data, generate_drifted_data
from src.data.preprocessor import validate_schema, split_data, prepare_features, compute_feature_stats
from src.features.store import FeatureStoreManager
from src.training.trainer import run_hyperparameter_sweep
from src.training.evaluator import evaluate_model, generate_evaluation_report
from src.drift.detector import DriftDetector
from src.pipeline.promoter import ModelPromoter
from src.serving.model_loader import ModelRegistry
from src.monitoring.metrics import (
    RETRAIN_EVENTS, PIPELINE_HEALTH, LAST_DRIFT_CHECK,
    LAST_RETRAIN_TIMESTAMP, update_drift_metrics, update_model_metrics,
    FEATURE_MEAN, FEATURE_STD,
)

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    End-to-end self-healing ML pipeline.

    Lifecycle:
      bootstrap() → initial data gen, feature store, training, promotion
      run_drift_monitor() → continuous loop checking for drift and healing
    """

    def __init__(self):
        self.data_dir = DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        self.registry = ModelRegistry()
        self.promoter = ModelPromoter(self.registry)
        self.feature_store = FeatureStoreManager()
        self.drift_detector: Optional[DriftDetector] = None

        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._train_stats: Optional[dict] = None
        self._retrain_count = 0
        self._running = False

    def bootstrap(self):
        """
        Run the full initial pipeline: data → features → train → evaluate → promote.
        """
        logger.info("=" * 60)
        logger.info("PIPELINE BOOTSTRAP")
        logger.info("=" * 60)

        # step 1: generate training data
        logger.info("[1/5] Generating training data...")
        raw_data = generate_training_data(
            n_samples=NUM_TRAIN_SAMPLES,
            fraud_ratio=FRAUD_RATIO,
            seed=RANDOM_SEED,
        )
        raw_data = validate_schema(raw_data)

        # save raw data
        raw_path = self.data_dir / "training_data.parquet"
        raw_data.to_parquet(raw_path, index=False)
        logger.info(f"Generated {len(raw_data)} samples, fraud rate: {raw_data[TARGET_COLUMN].mean():.3f}")

        # step 2: feature store
        logger.info("[2/5] Initializing feature store...")
        self.feature_store.initialize(raw_data)

        # point-in-time feature retrieval
        entity_df = raw_data[["entity_id", "event_timestamp"]].copy()
        feature_df = self.feature_store.get_historical_features(entity_df)

        # merge back the target
        feature_df[TARGET_COLUMN] = raw_data[TARGET_COLUMN].values

        # step 3: split and train
        logger.info("[3/5] Splitting data and running training sweep...")
        self._train_df, self._test_df = split_data(feature_df)
        self._train_stats = compute_feature_stats(self._train_df)

        X_train, y_train = prepare_features(self._train_df)
        X_test, y_test = prepare_features(self._test_df)

        sweep_results = run_hyperparameter_sweep(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        if not sweep_results:
            raise RuntimeError("All model training attempts failed")

        # step 4: evaluate best candidate
        logger.info("[4/5] Evaluating best candidate...")
        best_name, best_params, best_model, best_metrics, best_run_id = sweep_results[0]

        report = generate_evaluation_report(best_model, X_test, y_test, model_name=best_name)
        logger.info(f"\n{report}")

        # step 5: promote
        logger.info("[5/5] Attempting model promotion...")
        promoted, version, verdict = self.promoter.attempt_promotion(
            candidate_model=best_model,
            candidate_name=best_name,
            candidate_params=best_params,
            candidate_metrics=best_metrics,
            X_holdout=X_test,
            y_holdout=y_test,
            mlflow_run_id=best_run_id,
        )

        if not promoted:
            # in bootstrap, force-promote the best we have
            logger.warning("Candidate didn't pass gates, force-promoting for bootstrap")
            self.registry.promote(version)

        # set up drift detector against training distribution
        self.drift_detector = DriftDetector(self._train_df)

        # push feature stats to prometheus
        self._push_feature_stats(self._train_df)

        PIPELINE_HEALTH.set(1)
        logger.info("=" * 60)
        logger.info(f"BOOTSTRAP COMPLETE — serving model {version}")
        logger.info("=" * 60)

        return version

    def check_and_heal(self, incoming_data: Optional[pd.DataFrame] = None) -> dict:
        """
        Single iteration of the self-healing loop.

        1. Get new data (or simulate it)
        2. Check for drift
        3. If drift detected, retrain and promote
        """
        if self.drift_detector is None:
            raise RuntimeError("Pipeline not bootstrapped. Call bootstrap() first.")

        # simulate incoming production data with potential drift
        if incoming_data is None:
            drift_intensity = 0.5 + (self._retrain_count * 0.4)
            incoming_data = generate_drifted_data(
                n_samples=NUM_DRIFT_SAMPLES,
                fraud_ratio=FRAUD_RATIO,
                seed=int(time.time()) % 10000,
                drift_intensity=min(drift_intensity, 3.0),
            )
            incoming_data = validate_schema(incoming_data)

        # drift check
        drift_report = self.drift_detector.check(incoming_data)
        update_drift_metrics(drift_report)
        LAST_DRIFT_CHECK.set(time.time())

        logger.info(drift_report.summary())

        result = {
            "drift_report": drift_report,
            "retrained": False,
            "promoted": False,
            "new_version": None,
        }

        if not drift_report.should_retrain:
            return result

        # SELF-HEALING: retrain on combined old + new data
        logger.info(">>> DRIFT DETECTED — initiating self-healing retrain <<<")
        RETRAIN_EVENTS.labels(trigger_reason="psi_drift").inc()
        LAST_RETRAIN_TIMESTAMP.set(time.time())
        self._retrain_count += 1

        # combine training data with drifted data for retraining
        combined = pd.concat([self._train_df, incoming_data], ignore_index=True)
        combined = validate_schema(combined)

        train_new, test_new = split_data(combined)
        X_train, y_train = prepare_features(train_new)
        X_test, y_test = prepare_features(test_new)

        # retrain — use a focused sweep on the top architectures
        sweep_results = run_hyperparameter_sweep(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_names=["lightgbm", "xgboost", "gradient_boosting"],
        )

        if not sweep_results:
            logger.error("Retraining failed — no successful model runs")
            return result

        best_name, best_params, best_model, best_metrics, best_run_id = sweep_results[0]

        promoted, version, verdict = self.promoter.attempt_promotion(
            candidate_model=best_model,
            candidate_name=best_name,
            candidate_params=best_params,
            candidate_metrics=best_metrics,
            X_holdout=X_test,
            y_holdout=y_test,
            mlflow_run_id=best_run_id,
        )

        if promoted:
            # update reference distribution for future drift checks
            self._train_df = train_new
            self._train_stats = compute_feature_stats(train_new)
            self.drift_detector = DriftDetector(train_new)
            self._push_feature_stats(train_new)

        result["retrained"] = True
        result["promoted"] = promoted
        result["new_version"] = version
        return result

    def run_drift_monitor(self, interval_seconds: int = 30, max_iterations: int = 0):
        """
        Continuous drift monitoring loop.

        max_iterations=0 means run forever.
        """
        self._running = True
        iteration = 0

        logger.info(f"Starting drift monitor (interval={interval_seconds}s)")

        while self._running:
            iteration += 1
            if max_iterations > 0 and iteration > max_iterations:
                break

            logger.info(f"\n--- Drift check #{iteration} ---")
            try:
                result = self.check_and_heal()
                if result["promoted"]:
                    logger.info(f"Model updated to {result['new_version']} after healing")
            except Exception as e:
                logger.error(f"Drift check failed: {e}", exc_info=True)
                PIPELINE_HEALTH.set(0)

            if self._running:
                time.sleep(interval_seconds)

        logger.info("Drift monitor stopped.")

    def stop(self):
        self._running = False

    def _push_feature_stats(self, df: pd.DataFrame):
        """Update Prometheus gauges with current feature distributions."""
        for col in FEATURE_COLUMNS:
            if col in df.columns:
                FEATURE_MEAN.labels(feature_name=col).set(float(df[col].mean()))
                FEATURE_STD.labels(feature_name=col).set(float(df[col].std()))
