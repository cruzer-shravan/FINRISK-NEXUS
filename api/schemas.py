"""Pydantic schemas for FINRISK NEXUS API."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Batch prediction request."""

    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


class PredictionItem(BaseModel):
    """Single prediction response."""

    risk_probability: float = Field(ge=0.0, le=1.0)
    predicted_class: int = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0.0, le=1.0)
    model_name: str


class PredictionResponse(BaseModel):
    """Batch prediction response."""

    dataset: str
    count: int
    predictions: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    service: str


class ModelInfo(BaseModel):
    dataset: str
    model_name: str
    feature_count: int
    decision_threshold: float


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
