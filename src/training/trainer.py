"""
Model training with MLflow experiment tracking.

Trains multiple classifiers with cross-validation and hyperparameter
sweeps, logs everything to MLflow, and returns the best candidate
based on precision-recall tradeoffs.
"""

import logging
import time
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
    precision_recall_curve,
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from typing import Dict, Any, Tuple, Optional

from config.settings import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    RANDOM_SEED, CV_FOLDS, FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


# candidate architectures with their sweep grids
MODEL_REGISTRY = {
    "logistic_regression": {
        "class": LogisticRegression,
        "param_grid": [
            {"C": 0.01, "penalty": "l2", "max_iter": 500, "solver": "lbfgs"},
            {"C": 0.1, "penalty": "l2", "max_iter": 500, "solver": "lbfgs"},
            {"C": 1.0, "penalty": "l2", "max_iter": 500, "solver": "lbfgs"},
            {"C": 10.0, "penalty": "l2", "max_iter": 500, "solver": "lbfgs"},
        ],
    },
    "random_forest": {
        "class": RandomForestClassifier,
        "param_grid": [
            {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5},
            {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 3},
            {"n_estimators": 300, "max_depth": 16, "min_samples_leaf": 2},
        ],
    },
    "gradient_boosting": {
        "class": GradientBoostingClassifier,
        "param_grid": [
            {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8},
            {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9},
            {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.85},
        ],
    },
    "lightgbm": {
        "class": LGBMClassifier,
        "param_grid": [
            {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.1,
             "num_leaves": 31, "min_child_samples": 20, "verbose": -1},
            {"n_estimators": 250, "max_depth": 8, "learning_rate": 0.05,
             "num_leaves": 63, "min_child_samples": 10, "verbose": -1},
            {"n_estimators": 400, "max_depth": 10, "learning_rate": 0.03,
             "num_leaves": 127, "min_child_samples": 5, "verbose": -1},
        ],
    },
    "xgboost": {
        "class": XGBClassifier,
        "param_grid": [
            {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.1,
             "subsample": 0.8, "colsample_bytree": 0.8,
             "eval_metric": "logloss", "verbosity": 0, "use_label_encoder": False},
            {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.05,
             "subsample": 0.9, "colsample_bytree": 0.85,
             "eval_metric": "logloss", "verbosity": 0, "use_label_encoder": False},
        ],
    },
}


def _setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def _cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = CV_FOLDS,
) -> Dict[str, float]:
    """Run stratified k-fold CV and return averaged metrics."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "avg_precision": "average_precision",
    }
    cv_results = cross_validate(
        model, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )
    return {
        metric: float(np.mean(cv_results[f"test_{metric}"]))
        for metric in scoring
    }


def train_single_model(
    model_name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_tag: str = "sweep",
) -> Tuple[Any, Dict[str, float], str]:
    """
    Train one model config, log to MLflow, return (model, metrics, run_id).
    """
    _setup_mlflow()

    model_info = MODEL_REGISTRY[model_name]
    model_cls = model_info["class"]

    full_params = {**params, "random_state": RANDOM_SEED}
    # not all models accept random_state
    try:
        model = model_cls(**full_params)
    except TypeError:
        full_params.pop("random_state", None)
        model = model_cls(**params)

    with mlflow.start_run(run_name=f"{model_name}_{run_tag}") as run:
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("run_tag", run_tag)
        mlflow.log_params(params)

        # cross-validation metrics
        cv_metrics = _cross_validate_model(model, X_train, y_train)
        for k, v in cv_metrics.items():
            mlflow.log_metric(f"cv_{k}", v)

        # fit on full training set
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("training_time_sec", train_time)

        # test-set evaluation
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        test_metrics = {
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_prob),
            "test_avg_precision": average_precision_score(y_test, y_prob),
        }
        mlflow.log_metrics(test_metrics)

        # log precision-recall curve data as artifact
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        pr_data = {
            "precisions": precisions.tolist(),
            "recalls": recalls.tolist(),
            "thresholds": thresholds.tolist(),
        }
        mlflow.log_dict(pr_data, "pr_curve.json")

        # log the model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_test.iloc[:3],
        )

        all_metrics = {**cv_metrics, **test_metrics}
        logger.info(
            f"  {model_name} | F1={test_metrics['test_f1']:.4f} "
            f"P={test_metrics['test_precision']:.4f} R={test_metrics['test_recall']:.4f} "
            f"AUC={test_metrics['test_roc_auc']:.4f}"
        )

        return model, all_metrics, run.info.run_id


def run_hyperparameter_sweep(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_names: Optional[list] = None,
) -> list:
    """
    Run the full sweep across all model architectures and their param grids.
    Returns a list of (model_name, params, model, metrics, run_id) sorted by test_f1.
    """
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    _setup_mlflow()
    results = []

    for model_name in model_names:
        info = MODEL_REGISTRY[model_name]
        logger.info(f"Running sweep for {model_name} ({len(info['param_grid'])} configs)...")

        for i, params in enumerate(info["param_grid"]):
            try:
                model, metrics, run_id = train_single_model(
                    model_name=model_name,
                    params=params,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    run_tag=f"sweep_{i}",
                )
                results.append((model_name, params, model, metrics, run_id))
            except Exception as e:
                logger.error(f"Failed {model_name} config {i}: {e}")

    # sort by test F1 descending
    results.sort(key=lambda r: r[3].get("test_f1", 0), reverse=True)

    if results:
        best = results[0]
        logger.info(
            f"\nBest model: {best[0]} | test_f1={best[3]['test_f1']:.4f} "
            f"| run_id={best[4]}"
        )

    return results
