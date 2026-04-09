"""
Feast feature store integration.

Handles materialization and point-in-time correct feature retrieval
to prevent training-serving skew. The offline store is file-based
for local dev; swap to BigQuery/Redshift for prod.
"""

import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from feast import FeatureStore, Entity, FeatureView, FileSource, Field
from feast.types import Float64, Int64, String
from feast.infra.offline_stores.file_source import FileSource as FeastFileSource

from config.settings import FEATURE_COLUMNS, FEATURE_REPO_DIR

logger = logging.getLogger(__name__)


class FeatureStoreManager:
    """
    Wraps Feast operations for the fraud pipeline.

    Point-in-time joins are critical here — we need to make sure that
    at training time, each sample only sees features that were available
    *before* the transaction occurred. Otherwise we leak future info
    and the model looks great in eval but falls apart in production.
    """

    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = Path(repo_path or FEATURE_REPO_DIR)
        self._store: Optional[FeatureStore] = None
        self._initialized = False

    def _ensure_repo(self, parquet_path: str):
        """Write the Feast repo config and feature definitions on the fly."""
        self.repo_path.mkdir(parents=True, exist_ok=True)

        feature_store_yaml = self.repo_path / "feature_store.yaml"
        feature_store_yaml.write_text(
            "project: fraud_detection\n"
            "registry: registry.db\n"
            "provider: local\n"
            "online_store:\n"
            "  type: sqlite\n"
            f"  path: {self.repo_path / 'online_store.db'}\n"
            "entity_key_serialization_version: 2\n"
        )

        # write feature definition module
        feature_def_path = self.repo_path / "features.py"
        feature_def_path.write_text(self._build_feature_definition(parquet_path))

        self._store = FeatureStore(repo_path=str(self.repo_path))
        self._initialized = True

    def _build_feature_definition(self, parquet_path: str) -> str:
        """Generate the Feast feature definition Python file."""
        parquet_path_escaped = parquet_path.replace("\\", "/")
        return f'''
from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float64, Int64

transaction_entity = Entity(
    name="entity_id",
    description="User/account identifier",
)

tx_source = FileSource(
    path="{parquet_path_escaped}",
    timestamp_field="event_timestamp",
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="tx_amount", dtype=Float64),
        Field(name="tx_hour", dtype=Int64),
        Field(name="tx_day_of_week", dtype=Int64),
        Field(name="merchant_category", dtype=Int64),
        Field(name="distance_from_home", dtype=Float64),
        Field(name="distance_from_last_tx", dtype=Float64),
        Field(name="ratio_to_median_price", dtype=Float64),
        Field(name="is_chip_used", dtype=Int64),
        Field(name="is_pin_used", dtype=Int64),
        Field(name="is_online", dtype=Int64),
        Field(name="tx_frequency_1h", dtype=Float64),
        Field(name="tx_amount_avg_7d", dtype=Float64),
        Field(name="tx_amount_std_7d", dtype=Float64),
    ],
    source=tx_source,
    online=True,
)
'''

    def initialize(self, training_data: pd.DataFrame):
        """Set up the feature store with training data."""
        data_dir = self.repo_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = str(data_dir / "transactions.parquet")
        training_data.to_parquet(parquet_path, index=False)

        self._ensure_repo(parquet_path)

        logger.info("Applying Feast feature definitions...")
        self._store.apply(
            objects=self._load_feature_objects(),
            partial=False,
        )
        logger.info("Feast feature store initialized.")

    def _load_feature_objects(self):
        """Import and return the feature objects from the generated module."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "feast_features",
            str(self.repo_path / "features.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return [mod.transaction_entity, mod.transaction_features]

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Point-in-time feature retrieval.

        entity_df needs: entity_id, event_timestamp
        Returns the joined feature dataframe with only features available
        at or before each entity's timestamp.
        """
        if not self._initialized:
            raise RuntimeError("Feature store not initialized. Call initialize() first.")

        if "event_timestamp" not in entity_df.columns:
            raise ValueError("entity_df must contain 'event_timestamp'")

        feature_refs = [
            f"transaction_features:{col}" for col in FEATURE_COLUMNS
        ]

        logger.info(f"Retrieving historical features for {len(entity_df)} entities...")
        job = self._store.get_historical_features(
            entity_df=entity_df[["entity_id", "event_timestamp"]],
            features=feature_refs,
        )
        return job.to_df()

    def materialize(self, start_date: datetime, end_date: datetime):
        """Push features to the online store for serving."""
        if not self._initialized:
            raise RuntimeError("Feature store not initialized.")

        logger.info(f"Materializing features from {start_date} to {end_date}...")
        self._store.materialize(
            start_date=start_date,
            end_date=end_date,
        )
        logger.info("Materialization complete.")

    def get_online_features(self, entity_ids: list) -> pd.DataFrame:
        """Fetch features from the online store for real-time serving."""
        if not self._initialized:
            raise RuntimeError("Feature store not initialized.")

        entity_rows = [{"entity_id": eid} for eid in entity_ids]
        feature_refs = [
            f"transaction_features:{col}" for col in FEATURE_COLUMNS
        ]
        result = self._store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        )
        return result.to_df()
