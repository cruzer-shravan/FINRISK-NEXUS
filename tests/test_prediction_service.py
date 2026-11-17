import numpy as np
import pandas as pd
import pytest
import src.inference.prediction_service as service_module


class FakeModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        probabilities = np.linspace(0.1, 0.9, len(frame))
        return np.column_stack(
            [
                1 - probabilities,
                probabilities,
            ]
        )


def paysim_contract():
    return {
        "model_name": "test_paysim",
        "feature_order": [
            "amount",
            "oldbalanceOrg",
            "oldbalanceDest",
            "log_amount",
            "type",
        ],
        "numeric_features": [
            "amount",
            "oldbalanceOrg",
            "oldbalanceDest",
            "log_amount",
        ],
        "categorical_features": ["type"],
        "decision_threshold": 0.5,
    }


def home_credit_contract():
    return {
        "model_name": "test_home_credit",
        "feature_order": [
            "AMT_INCOME_TOTAL",
            "CODE_GENDER",
        ],
        "numeric_features": ["AMT_INCOME_TOTAL"],
        "categorical_features": ["CODE_GENDER"],
        "decision_threshold": 0.5,
    }


@pytest.fixture
def patch_loader(monkeypatch):
    def apply(contract):
        monkeypatch.setattr(
            service_module,
            "load_model_bundle",
            lambda path: (contract, FakeModel()),
        )

    return apply


def test_rejects_unsupported_dataset():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        service_module.PredictionService("unknown")


def test_paysim_prediction_output(patch_loader):
    patch_loader(paysim_contract())

    frame = pd.DataFrame(
        {
            "amount": [100.0, 200.0, 300.0],
            "oldbalanceOrg": [500.0, 600.0, 700.0],
            "oldbalanceDest": [10.0, 20.0, 30.0],
            "log_amount": np.log1p([100.0, 200.0, 300.0]),
            "type": ["TRANSFER", "PAYMENT", "CASH_OUT"],
        }
    )

    result = service_module.PredictionService("paysim").predict(frame)

    assert len(result) == len(frame)
    assert list(result.columns) == [
        "risk_probability",
        "predicted_class",
        "decision_threshold",
        "model_name",
    ]
    assert result["risk_probability"].between(0, 1).all()
    assert set(result["predicted_class"].unique()).issubset({0, 1})
    assert (result["decision_threshold"] == 0.5).all()
    assert (result["model_name"] == "test_paysim").all()


def test_home_credit_prediction_preserves_identifier(patch_loader):
    patch_loader(home_credit_contract())

    frame = pd.DataFrame(
        {
            "SK_ID_CURR": [101, 102, 103],
            "CODE_GENDER": ["F", "M", "F"],
            "AMT_INCOME_TOTAL": [100000.0, 200000.0, 300000.0],
        }
    )

    result = service_module.PredictionService("home_credit").predict(frame)

    assert len(result) == 3
    assert result["SK_ID_CURR"].tolist() == [101, 102, 103]
    assert result["risk_probability"].between(0, 1).all()
    assert set(result["predicted_class"].unique()).issubset({0, 1})
    assert (result["decision_threshold"] == 0.5).all()
    assert (result["model_name"] == "test_home_credit").all()


def test_predictions_use_configured_threshold(patch_loader):
    patch_loader(paysim_contract())

    frame = pd.DataFrame(
        {
            "amount": [100.0, 200.0, 300.0],
            "oldbalanceOrg": [500.0, 600.0, 700.0],
            "oldbalanceDest": [10.0, 20.0, 30.0],
            "log_amount": np.log1p([100.0, 200.0, 300.0]),
            "type": ["TRANSFER", "PAYMENT", "CASH_OUT"],
        }
    )

    result = service_module.PredictionService("paysim").predict(frame)

    expected = (result["risk_probability"].to_numpy() >= 0.5).astype("int8")

    np.testing.assert_array_equal(
        result["predicted_class"],
        expected,
    )


def test_invalid_paysim_input_is_rejected(patch_loader):
    patch_loader(paysim_contract())

    frame = pd.DataFrame(
        {
            "amount": [100.0],
            "oldbalanceOrg": [500.0],
            "oldbalanceDest": [10.0],
            "log_amount": [999.0],
            "type": ["TRANSFER"],
        }
    )

    service = service_module.PredictionService("paysim")

    with pytest.raises(ValueError, match="inconsistent"):
        service.predict(frame)
