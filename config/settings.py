import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FEATURE_REPO_DIR = BASE_DIR / "config" / "feature_repo"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR / 'mlflow.db'}")
MLFLOW_EXPERIMENT_NAME = "fraud_detection_pipeline"

FEAST_REPO_PATH = str(FEATURE_REPO_DIR)

# model training
RANDOM_SEED = 42
TEST_SPLIT_RATIO = 0.2
CV_FOLDS = 5
TARGET_COLUMN = "is_fraud"

FEATURE_COLUMNS = [
    "tx_amount",
    "tx_hour",
    "tx_day_of_week",
    "merchant_category",
    "distance_from_home",
    "distance_from_last_tx",
    "ratio_to_median_price",
    "is_chip_used",
    "is_pin_used",
    "is_online",
    "tx_frequency_1h",
    "tx_amount_avg_7d",
    "tx_amount_std_7d",
]

# drift detection thresholds
PSI_THRESHOLD = 0.15       # trigger alert
PSI_RETRAIN_THRESHOLD = 0.25  # trigger automatic retraining
PSI_NUM_BINS = 10

# model promotion
MIN_PRECISION = 0.70
MIN_RECALL = 0.55
MIN_F1_IMPROVEMENT = 0.005  # new model must beat current by at least this

# serving
SERVING_HOST = os.getenv("SERVING_HOST", "0.0.0.0")
SERVING_PORT = int(os.getenv("SERVING_PORT", "8000"))
MODEL_REGISTRY_PATH = ARTIFACTS_DIR / "registry"

# monitoring
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
PREDICTION_LOG_BUFFER_SIZE = 500

# data generation
NUM_TRAIN_SAMPLES = 15000
NUM_DRIFT_SAMPLES = 3000
FRAUD_RATIO = 0.035  # ~3.5% fraud rate, realistic
