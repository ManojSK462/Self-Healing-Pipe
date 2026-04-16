"""
Model registry and loader.

Manages the lifecycle of trained models — saving, loading, versioning,
and swapping the active model at serving time. Uses the local filesystem
as the model registry; in prod this would be S3/GCS + MLflow Model Registry.
"""

import json
import logging
import joblib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from threading import Lock

from config.settings import MODEL_REGISTRY_PATH

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Local model registry with versioning and atomic swaps.

    Directory structure:
      registry/
        models/
          v1/
            model.joblib
            metadata.json
          v2/
            model.joblib
            metadata.json
        active.json  <-- points to the current version
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.base_path = registry_path or MODEL_REGISTRY_PATH
        self.models_dir = self.base_path / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._active_model = None
        self._active_version = None
        self._active_metadata = None
        self._lock = Lock()

    def save_model(
        self,
        model: Any,
        metrics: Dict[str, float],
        model_name: str,
        params: dict,
        mlflow_run_id: Optional[str] = None,
    ) -> str:
        """Save a model to the registry. Returns the version string."""
        version = self._next_version()
        version_dir = self.models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "model.joblib"
        joblib.dump(model, model_path)

        metadata = {
            "version": version,
            "model_name": model_name,
            "params": params,
            "metrics": metrics,
            "mlflow_run_id": mlflow_run_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        meta_path = version_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        logger.info(f"Saved model {model_name} as {version}")
        return version

    def promote(self, version: str):
        """Set a version as the active serving model."""
        version_dir = self.models_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Version {version} not found in registry")

        active_ref = self.base_path / "active.json"
        active_ref.write_text(json.dumps({"active_version": version}))

        # reload into memory
        with self._lock:
            self._load_version(version)

        logger.info(f"Promoted {version} as active model")

    def load_active(self) -> Tuple[Any, str, Dict]:
        """Load the currently active model. Returns (model, version, metadata)."""
        with self._lock:
            if self._active_model is not None:
                return self._active_model, self._active_version, self._active_metadata

        active_ref = self.base_path / "active.json"
        if not active_ref.exists():
            raise FileNotFoundError("No active model set. Train and promote a model first.")

        ref = json.loads(active_ref.read_text())
        version = ref["active_version"]

        with self._lock:
            self._load_version(version)
            return self._active_model, self._active_version, self._active_metadata

    def _load_version(self, version: str):
        """Internal: load a specific version into memory."""
        version_dir = self.models_dir / version
        model_path = version_dir / "model.joblib"
        meta_path = version_dir / "metadata.json"

        self._active_model = joblib.load(model_path)
        self._active_version = version
        self._active_metadata = json.loads(meta_path.read_text())
        logger.info(f"Loaded model {version} into memory")

    def _next_version(self) -> str:
        """Generate the next version number."""
        existing = [
            d.name for d in self.models_dir.iterdir()
            if d.is_dir() and d.name.startswith("v")
        ]
        if not existing:
            return "v1"
        nums = []
        for v in existing:
            try:
                nums.append(int(v[1:]))
            except ValueError:
                pass
        return f"v{max(nums) + 1}" if nums else "v1"

    def get_active_version(self) -> Optional[str]:
        active_ref = self.base_path / "active.json"
        if not active_ref.exists():
            return None
        ref = json.loads(active_ref.read_text())
        return ref.get("active_version")

    def get_metadata(self, version: str) -> Dict:
        meta_path = self.models_dir / version / "metadata.json"
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text())

    def list_versions(self) -> list:
        versions = []
        for d in sorted(self.models_dir.iterdir()):
            if d.is_dir() and d.name.startswith("v"):
                meta = self.get_metadata(d.name)
                versions.append(meta)
        return versions

    def reload_active(self):
        """Force-reload the active model from disk (after a promotion)."""
        with self._lock:
            self._active_model = None
            self._active_version = None
            self._active_metadata = None
        self.load_active()
