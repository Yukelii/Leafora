"""Leafora FastAPI application.

Startup behaviour:
    * build the model, class mapping and sensor services once
    * load the Keras model once (never per request)
    * leave the API usable when the model is missing: `/api/health` reports it
      and `/api/predict` answers 503 instead of crashing the process
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import diagnosis, health, predict, sensors
from app.api.errors import install_error_handlers
from app.config import settings
from app.decision.diagnosis import DiagnosisEngine, DiagnosisSettings
from app.ml.inference import ModelService
from app.services.prediction_service import PredictionService
from app.services.sensor_service import SensorService

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_model_service() -> ModelService:
    return ModelService(
        model_path=settings.model_path,
        class_names_path=settings.class_names_path,
        model_version=settings.model_version,
        image_size=settings.image_size,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_service: ModelService = app.state.model_service
    logger.info("Starting Leafora API %s", settings.model_version)
    model_service.load()
    if not model_service.is_loaded:
        logger.warning(
            "Inference disabled: /api/predict will return 503 until a model is exported "
            "and MODEL_PATH points at it."
        )
    yield
    model_service.unload()
    logger.info("Leafora API stopped")


def create_app() -> FastAPI:
    """Application factory, so tests can build an app with a stub model."""
    app = FastAPI(
        title="Leafora API",
        version="1.0.0",
        description=(
            "Tomato leaf disease prediction and ESP32 environmental sensor intake. "
            "OpenAPI docs stay enabled at /docs and /openapi.json."
        ),
        lifespan=lifespan,
    )

    # Services live on app.state: created once, shared by every request.
    app.state.model_service = build_model_service()
    app.state.prediction_service = PredictionService(app.state.model_service)
    app.state.sensor_service = SensorService(history_limit=settings.sensor_history_limit)
    # Rule-based layer: separate from the classifier on purpose. It never touches
    # the image and never changes a probability.
    app.state.diagnosis_engine = DiagnosisEngine(
        DiagnosisSettings(
            image_weight=settings.diagnosis_image_weight,
            symptom_weight=settings.diagnosis_symptom_weight,
            min_confidence=settings.diagnosis_min_confidence,
            min_margin=settings.diagnosis_min_margin,
            climate_weight=settings.diagnosis_climate_weight,
            climate_bonus=settings.diagnosis_climate_bonus,
        )
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    install_error_handlers(app)

    app.include_router(health.router, prefix="/api")
    app.include_router(predict.router, prefix="/api")
    app.include_router(diagnosis.router, prefix="/api")
    app.include_router(sensors.router, prefix="/api")

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": "Leafora API", "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
