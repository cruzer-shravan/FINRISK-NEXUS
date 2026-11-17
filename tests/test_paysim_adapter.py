import numpy as np
import pandas as pd
import pytest
from src.inference.paysim_adapter import prepare_paysim_features

CONTRACT = {
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
}


@pytest.fixture
def valid_frame():
    return pd.DataFrame(
        {
            "amount": [100.0, 250.0],
            "oldbalanceOrg": [500.0, 800.0],
            "oldbalanceDest": [20.0, 50.0],
            "log_amount": np.log1p([100.0, 250.0]),
            "type": ["TRANSFER", "PAYMENT"],
        }
    )


def test_accepts_valid_input(valid_frame):
    result = prepare_paysim_features(valid_frame, CONTRACT)

    assert list(result.columns) == CONTRACT["feature_order"]
    assert len(result) == 2

    for column in CONTRACT["numeric_features"]:
        assert result[column].dtype == "float64"


def test_rejects_non_dataframe():
    with pytest.raises(TypeError):
        prepare_paysim_features([1, 2], CONTRACT)


def test_rejects_empty_input():
    frame = pd.DataFrame(columns=CONTRACT["feature_order"])

    with pytest.raises(ValueError, match="no rows"):
        prepare_paysim_features(frame, CONTRACT)


def test_rejects_missing_feature(valid_frame):
    frame = valid_frame.drop(columns=["amount"])

    with pytest.raises(ValueError, match="Missing"):
        prepare_paysim_features(frame, CONTRACT)


def test_rejects_unexpected_column(valid_frame):
    frame = valid_frame.assign(unexpected=1)

    with pytest.raises(ValueError, match="Unexpected"):
        prepare_paysim_features(frame, CONTRACT)


def test_rejects_missing_values(valid_frame):
    frame = valid_frame.copy()
    frame.loc[0, "amount"] = np.nan
    frame.loc[0, "log_amount"] = np.nan

    with pytest.raises(ValueError, match="missing"):
        prepare_paysim_features(frame, CONTRACT)


def test_rejects_inconsistent_log_amount(valid_frame):
    frame = valid_frame.copy()
    frame.loc[0, "log_amount"] = 999.0

    with pytest.raises(ValueError, match="inconsistent"):
        prepare_paysim_features(frame, CONTRACT)


@pytest.mark.parametrize(
    "column",
    ["amount", "oldbalanceOrg", "oldbalanceDest"],
)
def test_rejects_negative_amount_or_balance(valid_frame, column):
    frame = valid_frame.copy()
    frame.loc[0, column] = -1.0

    with pytest.raises(ValueError, match="negative"):
        prepare_paysim_features(frame, CONTRACT)


def test_rejects_unsupported_transaction_type(valid_frame):
    frame = valid_frame.copy()
    frame.loc[0, "type"] = "INVALID"

    with pytest.raises(ValueError, match="Unsupported"):
        prepare_paysim_features(frame, CONTRACT)


def test_rejects_duplicate_columns(valid_frame):
    frame = pd.concat(
        [valid_frame, valid_frame[["amount"]]],
        axis=1,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        prepare_paysim_features(frame, CONTRACT)
