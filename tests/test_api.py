"""Tests for FINRISK NEXUS FastAPI endpoints."""

import api.main as main_module
import pandas as pd
import pytest
from fastapi.testclient import TestClient


class FakePredictionService:
    """Lightweight prediction service for API tests."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.contract = {
            "model_name": (
                "paysim_random_forest" if dataset == "paysim" else "home_credit_calibrated_xgboost"
            ),
            "feature_count": 5 if dataset == "paysim" else 120,
            "decision_threshold": (
                0.9305254794485053 if dataset == "paysim" else 0.17248130501689904
            ),
        }

    def predict(self, frame):
        if self.dataset == "paysim":
            required = {
                "amount",
                "oldbalanceOrg",
                "oldbalanceDest",
                "log_amount",
                "type",
            }
            if set(frame.columns) != required:
                raise ValueError("Invalid PaySim feature set.")

        elif self.dataset == "home_credit":
            if "SK_ID_CURR" not in frame.columns:
                raise ValueError("SK_ID_CURR is required.")

            if "TARGET" in frame.columns:
                raise ValueError("TARGET is not accepted.")

        result = pd.DataFrame(
            {
                "risk_probability": [0.25] * len(frame),
                "predicted_class": [0] * len(frame),
                "decision_threshold": [self.contract["decision_threshold"]] * len(frame),
                "model_name": [self.contract["model_name"]] * len(frame),
            },
            index=frame.index,
        )

        if self.dataset == "home_credit":
            result.insert(0, "SK_ID_CURR", frame["SK_ID_CURR"])

        return result


@pytest.fixture
def client(monkeypatch):
    """Create a test client with mocked prediction services."""

    def fake_services():
        return {
            "paysim": FakePredictionService("paysim"),
            "home_credit": FakePredictionService("home_credit"),
        }

    monkeypatch.setattr(
        main_module,
        "get_prediction_services",
        fake_services,
    )

    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "FINRISK NEXUS",
    }


def test_models_endpoint(client):
    response = client.get("/models")

    assert response.status_code == 200

    models = response.json()["models"]

    assert len(models) == 2

    paysim = next(model for model in models if model["dataset"] == "paysim")
    home_credit = next(model for model in models if model["dataset"] == "home_credit")

    assert paysim["feature_count"] == 5
    assert home_credit["feature_count"] == 120


def test_paysim_prediction_endpoint(client):
    payload = {
        "records": [
            {
                "amount": 100.0,
                "oldbalanceOrg": 1000.0,
                "oldbalanceDest": 500.0,
                "log_amount": 4.6151205168,
                "type": "PAYMENT",
            }
        ]
    }

    response = client.post("/predict/paysim", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert body["dataset"] == "paysim"
    assert body["count"] == 1
    assert body["predictions"][0]["risk_probability"] == 0.25
    assert body["predictions"][0]["predicted_class"] == 0


def test_home_credit_prediction_preserves_identifier(client):
    payload = {
        "records": [
            {
                "SK_ID_CURR": 143234,
                "AMT_INCOME_TOTAL": 150000.0,
            }
        ]
    }

    response = client.post("/predict/home-credit", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert body["dataset"] == "home_credit"
    assert body["count"] == 1
    assert body["predictions"][0]["SK_ID_CURR"] == 143234


def test_rejects_empty_records(client):
    response = client.post(
        "/predict/paysim",
        json={"records": []},
    )

    assert response.status_code == 422


def test_rejects_missing_records(client):
    response = client.post(
        "/predict/paysim",
        json={},
    )

    assert response.status_code == 422


def test_rejects_unexpected_request_field(client):
    response = client.post(
        "/predict/paysim",
        json={
            "records": [{"amount": 100.0}],
            "unexpected": True,
        },
    )

    assert response.status_code == 422


def test_rejects_invalid_paysim_features(client):
    response = client.post(
        "/predict/paysim",
        json={
            "records": [
                {
                    "amount": 100.0,
                    "oldbalanceOrg": 1000.0,
                }
            ]
        },
    )

    assert response.status_code == 422


def test_rejects_home_credit_target(client):
    response = client.post(
        "/predict/home-credit",
        json={
            "records": [
                {
                    "SK_ID_CURR": 143234,
                    "TARGET": 1,
                }
            ]
        },
    )

    assert response.status_code == 422


def test_openapi_endpoint(client):
    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/health" in paths
    assert "/models" in paths
    assert "/predict/paysim" in paths
    assert "/predict/home-credit" in paths
