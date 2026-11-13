"""GET /api/health - liveness plus honest model status."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_model_service
from app.ml.inference import ModelService

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool = Field(..., examples=[True])
    model_version: str = Field(..., examples=["leafora-densenet121-v1"])
    num_classes: int = Field(..., examples=[10])
    sensor_buffer_count: int = Field(..., examples=[3])
    detail: str | None = Field(default=None, examples=[None])


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health(
    model_service: ModelService = Depends(get_model_service),
) -> HealthResponse:
    """Report whether the API is up and whether the model is loaded.

    `model_loaded` is true only when a model object *and* its class mapping are
    both in memory, so a broken model export cannot look healthy.
    """
    loaded = model_service.is_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=model_service.model_version,
        num_classes=model_service.num_classes,
        sensor_buffer_count=0,
        detail=model_service.load_error,
    )
