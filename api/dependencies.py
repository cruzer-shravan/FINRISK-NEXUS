"""Shared API dependencies and prediction service initialization."""

from functools import lru_cache

from src.inference.prediction_service import PredictionService


@lru_cache(maxsize=1)
def get_prediction_services() -> dict[str, PredictionService]:
    """Load and cache the frozen prediction services."""
    return {
        "paysim": PredictionService("paysim"),
        "home_credit": PredictionService("home_credit"),
    }
