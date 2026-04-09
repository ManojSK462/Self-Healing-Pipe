"""
Prometheus metrics for the ML pipeline.

Tracks prediction latency, drift scores, model performance,
retraining events, and feature distribution stats. These feed
into the Grafana dashboards.
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
)

# dedicated registry so we don't collide with default process metrics
REGISTRY = CollectorRegistry(auto_describe=True)

# -- prediction serving metrics --

PREDICTION_COUNT = Counter(
    "ml_predictions_total",
    "Total number of predictions served",
    ["model_version", "predicted_class"],
    registry=REGISTRY,
)

PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Prediction serving latency",
    ["model_version"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY,
)

PREDICTION_PROBABILITY = Histogram(
    "ml_prediction_probability",
    "Distribution of fraud probability scores",
    ["model_version"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY,
)

# -- drift monitoring --

FEATURE_PSI = Gauge(
    "ml_feature_psi",
    "PSI score per feature",
    ["feature_name"],
    registry=REGISTRY,
)

AGGREGATE_PSI = Gauge(
    "ml_aggregate_psi",
    "Aggregate PSI across all features",
    registry=REGISTRY,
)

DRIFT_ALERT_LEVEL = Gauge(
    "ml_drift_alert_level",
    "Drift alert level (0=none, 1=warning, 2=critical)",
    registry=REGISTRY,
)

# -- model lifecycle --

MODEL_F1_SCORE = Gauge(
    "ml_model_f1_score",
    "Current model F1 score",
    ["model_version", "dataset"],
    registry=REGISTRY,
)

MODEL_PRECISION = Gauge(
    "ml_model_precision",
    "Current model precision",
    ["model_version", "dataset"],
    registry=REGISTRY,
)

MODEL_RECALL = Gauge(
    "ml_model_recall",
    "Current model recall",
    ["model_version", "dataset"],
    registry=REGISTRY,
)

MODEL_AUC = Gauge(
    "ml_model_auc",
    "Current model ROC AUC",
    ["model_version", "dataset"],
    registry=REGISTRY,
)

RETRAIN_EVENTS = Counter(
    "ml_retrain_events_total",
    "Number of automated retraining events",
    ["trigger_reason"],
    registry=REGISTRY,
)

ACTIVE_MODEL_INFO = Info(
    "ml_active_model",
    "Currently active model metadata",
    registry=REGISTRY,
)

PROMOTION_EVENTS = Counter(
    "ml_promotion_events_total",
    "Number of model promotion events",
    registry=REGISTRY,
)

# -- feature distribution tracking --

FEATURE_MEAN = Gauge(
    "ml_feature_mean",
    "Current mean of each feature in production data",
    ["feature_name"],
    registry=REGISTRY,
)

FEATURE_STD = Gauge(
    "ml_feature_std",
    "Current std of each feature in production data",
    ["feature_name"],
    registry=REGISTRY,
)

# -- system health --

PIPELINE_HEALTH = Gauge(
    "ml_pipeline_health",
    "Pipeline health status (1=healthy, 0=degraded)",
    registry=REGISTRY,
)

LAST_DRIFT_CHECK = Gauge(
    "ml_last_drift_check_timestamp",
    "Unix timestamp of last drift check",
    registry=REGISTRY,
)

LAST_RETRAIN_TIMESTAMP = Gauge(
    "ml_last_retrain_timestamp",
    "Unix timestamp of last retraining run",
    registry=REGISTRY,
)


def get_metrics_output() -> bytes:
    """Serialize all registered metrics for the /metrics endpoint."""
    return generate_latest(REGISTRY)


def update_model_metrics(version: str, metrics: dict, dataset: str = "test"):
    """Convenience: push a metrics dict into the Prometheus gauges."""
    if f"{dataset}_f1" in metrics:
        MODEL_F1_SCORE.labels(model_version=version, dataset=dataset).set(
            metrics[f"{dataset}_f1"])
    if f"{dataset}_precision" in metrics:
        MODEL_PRECISION.labels(model_version=version, dataset=dataset).set(
            metrics[f"{dataset}_precision"])
    if f"{dataset}_recall" in metrics:
        MODEL_RECALL.labels(model_version=version, dataset=dataset).set(
            metrics[f"{dataset}_recall"])
    if f"{dataset}_roc_auc" in metrics:
        MODEL_AUC.labels(model_version=version, dataset=dataset).set(
            metrics[f"{dataset}_roc_auc"])


def update_drift_metrics(drift_report):
    """Push drift detection results into Prometheus gauges."""
    AGGREGATE_PSI.set(drift_report.aggregate_psi)

    alert_map = {"none": 0, "warning": 1, "critical": 2}
    DRIFT_ALERT_LEVEL.set(alert_map.get(drift_report.alert_level, 0))

    for feature, psi_val in drift_report.feature_psi.items():
        FEATURE_PSI.labels(feature_name=feature).set(psi_val)
