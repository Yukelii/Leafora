"""Pydantic schemas for prediction responses."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., examples=["Tomato___Early_blight"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.9731])
    model_version: str = Field(..., examples=["leafora-densenet121-v1"])
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        examples=["2025-01-15T08:30:00Z"],
    )


class PredictionDetailResponse(PredictionResponse):
    """Same as :class:`PredictionResponse` plus the full probability map."""

    probabilities: Dict[str, float]


class ErrorDetail(BaseModel):
    code: str = Field(..., examples=["INVALID_IMAGE"])
    message: str = Field(..., examples=["The uploaded image could not be processed."])


class ErrorResponse(BaseModel):
    error: ErrorDetail
