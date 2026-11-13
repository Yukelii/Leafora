"""Shared dependencies.

Services are created once in `app.main` and stored on `app.state`. These helpers
hand them to the route modules, which keeps the API layer free of construction
logic and makes the tests able to swap in a stub model.
"""

from __future__ import annotations

from fastapi import Request

from app.decision.diagnosis import DiagnosisEngine
from app.ml.inference import ModelService
from app.services.prediction_service import PredictionService
from app.services.sensor_service import SensorService


def get_prediction_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_sensor_service(request: Request) -> SensorService:
    return request.app.state.sensor_service


def get_diagnosis_engine(request: Request) -> DiagnosisEngine:
    return request.app.state.diagnosis_engine
