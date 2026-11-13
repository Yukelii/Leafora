"""Schemas for the rule-based diagnosis endpoint.

The response deliberately keeps the photo result, the symptom result and the
combined answer separate, so a user (or a panel) can see exactly why the system
said what it said.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RankedPrediction(BaseModel):
    predicted_class: str = Field(..., examples=["Tomato___Late_blight"])
    score: float = Field(..., ge=0.0, le=1.0, examples=[0.5123])


class PhotoPrediction(BaseModel):
    predicted_class: str = Field(..., examples=["Tomato___Late_blight"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.61])


class SymptomPrediction(BaseModel):
    predicted_class: str = Field(..., examples=["Tomato___Leaf_Mold"])
    score: float = Field(..., ge=0.0, le=1.0, examples=[0.42])


class CombineWeights(BaseModel):
    photo: float = Field(..., examples=[0.48])
    symptoms: float = Field(..., examples=[0.32])
    climate: Optional[float] = Field(
        default=None,
        description="Set only when sensor history supplied a climate prior.",
        examples=[0.2],
    )


class ClimateFeatures(BaseModel):
    sample_count: int = Field(..., examples=[48])
    window_hours: float = Field(..., examples=[11.5])
    mean_humidity: float = Field(..., examples=[91.2])
    wet_hours: float = Field(..., examples=[7.5])
    mean_temperature: float = Field(..., examples=[19.4])
    max_temperature: float = Field(..., examples=[23.1])
    min_temperature: float = Field(..., examples=[16.8])
    mean_soil_moisture: float = Field(..., examples=[68.4])
    moisture_trend: float = Field(..., examples=[-0.4])


class ClimatePrior(BaseModel):
    """The sensor-history prior, shown so the user can read the reasoning."""

    favoured_class: str = Field(..., examples=["Tomato___Late_blight"])
    support: float = Field(..., ge=0.0, le=1.0, examples=[0.72])
    features: ClimateFeatures
    reasons: List[str] = Field(default_factory=list)


class DiagnosisResponse(BaseModel):
    status: str = Field(
        ...,
        description="confident, uncertain or disagree",
        examples=["confident"],
    )
    predicted_class: Optional[str] = Field(
        default=None,
        description="Set only when status is 'confident'.",
        examples=["Tomato___Late_blight"],
    )
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.83])
    message: str = Field(
        ...,
        examples=["Most likely Late blight. The symptom checklist agrees."],
    )
    top_predictions: List[RankedPrediction] = Field(default_factory=list)
    photo_prediction: Optional[PhotoPrediction] = None
    symptom_prediction: Optional[SymptomPrediction] = None
    reasons: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(
        default_factory=list,
        description="What the rules are based on, so nothing is overclaimed.",
    )
    weights: CombineWeights
    climate: Optional[ClimatePrior] = Field(
        default=None,
        description="Sensor-history prior, when a device id was supplied.",
    )
    model_version: str = Field(..., examples=["leafora-densenet121-field-v1"])
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        examples=["2025-01-15T08:30:00Z"],
    )


class SymptomOption(BaseModel):
    key: str = Field(..., examples=["dark_spots"])
    label: str = Field(..., examples=["Dark or brown spots on the leaf"])


class SymptomQuestion(BaseModel):
    key: str = Field(..., examples=["damage_type"])
    prompt: str = Field(..., examples=["What does the main damage on the leaf look like?"])
    options: List[SymptomOption]


class SymptomChecklistResponse(BaseModel):
    """The questions the mobile app renders as tap buttons."""

    questions: List[SymptomQuestion]
    evidence: List[str] = Field(default_factory=list)
    symptom_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Always empty here; filled by POST /api/diagnose.",
    )
