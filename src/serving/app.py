"""
FastAPI prediction serving layer.

Exposes:
  POST /predict     — real-time fraud predictions
  GET  /health      — health check with model info
  GET  /metrics     — Prometheus metrics endpoint
  GET  /model/info  — active model metadata
"""

import logging
import time
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Optional

from src.serving.model_loader import ModelRegistry
from src.monitoring.metrics import (
    PREDICTION_COUNT, PREDICTION_LATENCY, PREDICTION_PROBABILITY,
    PIPELINE_HEALTH, ACTIVE_MODEL_INFO, get_metrics_output,
)
from config.settings import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model. Shutdown: cleanup."""
    try:
        model, version, meta = registry.load_active()
        ACTIVE_MODEL_INFO.info({
            "version": version,
            "model_name": meta.get("model_name", "unknown"),
            "created_at": meta.get("created_at", ""),
        })
        PIPELINE_HEALTH.set(1)
        logger.info(f"Serving started with model {version}")
    except FileNotFoundError:
        logger.warning("No active model found — server will start but /predict will 503")
        PIPELINE_HEALTH.set(0)
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Self-healing ML pipeline serving endpoint",
    version="1.0.0",
    lifespan=lifespan,
)


class TransactionInput(BaseModel):
    tx_amount: float = Field(..., gt=0)
    tx_hour: int = Field(..., ge=0, le=23)
    tx_day_of_week: int = Field(..., ge=0, le=6)
    merchant_category: int = Field(..., ge=0, le=7)
    distance_from_home: float = Field(..., ge=0)
    distance_from_last_tx: float = Field(..., ge=0)
    ratio_to_median_price: float = Field(..., ge=0)
    is_chip_used: int = Field(..., ge=0, le=1)
    is_pin_used: int = Field(..., ge=0, le=1)
    is_online: int = Field(..., ge=0, le=1)
    tx_frequency_1h: float = Field(..., ge=0)
    tx_amount_avg_7d: float = Field(..., ge=0)
    tx_amount_std_7d: float = Field(..., ge=0)


class PredictionRequest(BaseModel):
    transactions: List[TransactionInput]


class PredictionResult(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str


class PredictionResponse(BaseModel):
    model_version: str
    predictions: List[PredictionResult]
    latency_ms: float


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Score a batch of transactions for fraud."""
    try:
        model, version, _ = registry.load_active()
    except FileNotFoundError:
        raise HTTPException(503, "No model available. Pipeline has not completed initial training.")

    t0 = time.perf_counter()

    rows = [tx.model_dump() for tx in request.transactions]
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    probabilities = model.predict_proba(df)[:, 1]
    predictions_bin = model.predict(df)

    elapsed = (time.perf_counter() - t0) * 1000  # ms

    results = []
    for prob, pred in zip(probabilities, predictions_bin):
        if prob >= 0.8:
            risk = "high"
        elif prob >= 0.4:
            risk = "medium"
        else:
            risk = "low"

        results.append(PredictionResult(
            is_fraud=bool(pred),
            fraud_probability=round(float(prob), 4),
            risk_level=risk,
        ))

        # metrics
        PREDICTION_COUNT.labels(
            model_version=version,
            predicted_class="fraud" if pred else "legitimate",
        ).inc()
        PREDICTION_PROBABILITY.labels(model_version=version).observe(float(prob))

    PREDICTION_LATENCY.labels(model_version=version).observe(elapsed / 1000)

    return PredictionResponse(
        model_version=version,
        predictions=results,
        latency_ms=round(elapsed, 2),
    )


@app.get("/health")
async def health():
    try:
        _, version, meta = registry.load_active()
        return {
            "status": "healthy",
            "model_version": version,
            "model_name": meta.get("model_name"),
            "created_at": meta.get("created_at"),
        }
    except FileNotFoundError:
        return {"status": "no_model", "detail": "Awaiting initial training"}


@app.get("/model/info")
async def model_info():
    versions = registry.list_versions()
    active = registry.get_active_version()
    return {
        "active_version": active,
        "all_versions": versions,
    }


@app.get("/metrics")
async def metrics():
    return Response(
        content=get_metrics_output(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
