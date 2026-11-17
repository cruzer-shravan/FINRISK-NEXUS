"""FINRISK NEXUS FastAPI application."""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from api.dependencies import get_prediction_services
from api.schemas import (
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model services at application startup."""
    try:
        app.state.prediction_services = get_prediction_services()
        logger.info("Prediction services initialized successfully.")
    except Exception:
        logger.exception("Failed to initialize prediction services.")
        raise

    yield


app = FastAPI(
    title="FINRISK NEXUS API",
    description=("Financial risk prediction API for transaction fraud and credit default risk."),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status="healthy",
        service="FINRISK NEXUS",
    )


@app.get("/models", response_model=ModelsResponse, tags=["Models"])
def models(request: Request):
    services = request.app.state.prediction_services

    results = []

    for dataset, service in services.items():
        contract = service.contract

        results.append(
            ModelInfo(
                dataset=dataset,
                model_name=contract["model_name"],
                feature_count=contract["feature_count"],
                decision_threshold=contract["decision_threshold"],
            )
        )

    return ModelsResponse(models=results)


def run_prediction(
    request: Request,
    dataset: str,
    payload: PredictionRequest,
) -> PredictionResponse:
    """Execute batch inference through the shared prediction service."""

    services = request.app.state.prediction_services

    if dataset not in services:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported dataset: {dataset}",
        )

    try:
        frame = pd.DataFrame(payload.records)
        result = services[dataset].predict(frame)

        predictions = result.to_dict(orient="records")

        return PredictionResponse(
            dataset=dataset,
            count=len(predictions),
            predictions=predictions,
        )

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Prediction failed for dataset %s", dataset)

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal server error.",
        ) from exc


@app.post(
    "/predict/paysim",
    response_model=PredictionResponse,
    tags=["Predictions"],
)
def predict_paysim(
    payload: PredictionRequest,
    request: Request,
):
    return run_prediction(request, "paysim", payload)


@app.post(
    "/predict/home-credit",
    response_model=PredictionResponse,
    tags=["Predictions"],
)
def predict_home_credit(
    payload: PredictionRequest,
    request: Request,
):
    return run_prediction(request, "home_credit", payload)
