from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .config import settings
from .inference import load_predictor, predict_bytes
from .monitoring import PredictionStore


logging.basicConfig(
    level=settings.log_level,
    format="%(message)s",
)
logger = logging.getLogger("cats_dogs_api")

REQUEST_COUNT = Counter(
    "cats_dogs_requests_total",
    "Total HTTP requests",
    labelnames=("method", "endpoint", "status"),
)
REQUEST_LATENCY = Histogram(
    "cats_dogs_request_latency_seconds",
    "HTTP request latency in seconds",
    labelnames=("endpoint",),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTION_COUNT = Counter(
    "cats_dogs_predictions_total",
    "Total model predictions",
    labelnames=("label",),
)


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=8)
    true_label: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = load_predictor(settings.model_path)
    app.state.prediction_store = PredictionStore(settings.monitoring_db)
    logger.info(
        json.dumps(
            {
                "event": "startup",
                "model_path": str(settings.model_path),
                "architecture": app.state.model_bundle.architecture,
                "class_names": app.state.model_bundle.class_names,
            }
        )
    )
    yield
    logger.info(json.dumps({"event": "shutdown"}))


app = FastAPI(
    title="Cats vs Dogs Inference API",
    version="1.0.0",
    description="MLOps assignment inference service with monitoring and feedback-based post-deployment evaluation.",
    lifespan=lifespan,
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency = time.perf_counter() - started
        endpoint = request.url.path
        REQUEST_COUNT.labels(request.method, endpoint, "500").inc()
        REQUEST_LATENCY.labels(endpoint).observe(latency)
        logger.exception(
            json.dumps(
                {
                    "event": "request_error",
                    "request_id": request_id,
                    "method": request.method,
                    "path": endpoint,
                    "latency_ms": round(latency * 1000, 3),
                }
            )
        )
        raise
    latency = time.perf_counter() - started
    endpoint = request.url.path
    REQUEST_COUNT.labels(request.method, endpoint, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(endpoint).observe(latency)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        json.dumps(
            {
                "event": "request_complete",
                "request_id": request_id,
                "method": request.method,
                "path": endpoint,
                "status_code": response.status_code,
                "latency_ms": round(latency * 1000, 3),
            },
            sort_keys=True,
        )
    )
    return response


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    bundle = request.app.state.model_bundle
    return {
        "status": "healthy",
        "model_loaded": True,
        "architecture": bundle.architecture,
        "classes": bundle.class_names,
        "image_size": bundle.image_size,
    }


@app.get("/model-info")
def model_info(request: Request) -> dict[str, object]:
    bundle = request.app.state.model_bundle
    return {
        "architecture": bundle.architecture,
        "classes": bundle.class_names,
        "image_size": bundle.image_size,
        "metadata": bundle.metadata,
    }


@app.post("/predict")
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="JPEG/PNG image of a cat or dog")],
) -> JSONResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit")

    started = time.perf_counter()
    try:
        prediction = predict_bytes(data, request.app.state.model_bundle)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - started) * 1000
    request_id = str(uuid.uuid4())
    PREDICTION_COUNT.labels(prediction.label).inc()
    request.app.state.prediction_store.record_prediction(
        request_id=request_id,
        predicted_label=prediction.label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        latency_ms=latency_ms,
    )
    payload = {
        "request_id": request_id,
        "label": prediction.label,
        "confidence": round(prediction.confidence, 6),
        "probabilities": {key: round(value, 6) for key, value in prediction.probabilities.items()},
        "latency_ms": round(latency_ms, 3),
        "model_version": request.app.state.model_bundle.metadata.get("model_version", "unknown"),
    }
    return JSONResponse(payload)


@app.post("/feedback")
def feedback(body: FeedbackRequest, request: Request) -> dict[str, object]:
    allowed = set(request.app.state.model_bundle.class_names)
    if body.true_label not in allowed:
        raise HTTPException(status_code=422, detail=f"true_label must be one of {sorted(allowed)}")
    updated = request.app.state.prediction_store.add_feedback(body.request_id, body.true_label)
    if not updated:
        raise HTTPException(status_code=404, detail="request_id not found")
    return {"status": "recorded", "request_id": body.request_id, "true_label": body.true_label}


@app.get("/monitoring/performance")
def performance(request: Request) -> dict[str, object]:
    summary = request.app.state.prediction_store.performance()
    return {
        "labeled_count": summary.labeled_count,
        "correct_count": summary.correct_count,
        "accuracy": summary.accuracy,
        "true_label_distribution": summary.label_distribution,
        "prediction_distribution": summary.prediction_distribution,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
