import numpy as np
import pandas as pd
import pytest
from src.inference.home_credit_adapter import prepare_home_credit_features

CONTRACT = {
    "feature_order": [
        "AMT_INCOME_TOTAL",
        "CNT_CHILDREN",
        "CODE_GENDER",
    ],
    "numeric_features": [
        "AMT_INCOME_TOTAL",
        "CNT_CHILDREN",
    ],
    "categorical_features": [
        "CODE_GENDER",
    ],
}


@pytest.fixture
def valid_frame():
    return pd.DataFrame(
        {
            "CODE_GENDER": ["F", "M"],
            "CNT_CHILDREN": [0, 2],
            "AMT_INCOME_TOTAL": [150000.0, 250000.0],
            "SK_ID_CURR": [100001, 100002],
        }
    )


def test_preserves_exact_feature_order(valid_frame):
    features, _ = prepare_home_credit_features(valid_frame, CONTRACT)

    assert list(features.columns) == CONTRACT["feature_order"]


def test_retains_identifier_separately(valid_frame):
    features, identifiers = prepare_home_credit_features(valid_frame, CONTRACT)

    assert identifiers.tolist() == [100001, 100002]
    assert "SK_ID_CURR" not in features.columns


def test_normalizes_numeric_pd_na(valid_frame):
    frame = valid_frame.copy()
    frame["AMT_INCOME_TOTAL"] = frame["AMT_INCOME_TOTAL"].astype("object")
    frame.loc[0, "AMT_INCOME_TOTAL"] = pd.NA

    features, _ = prepare_home_credit_features(frame, CONTRACT)

    assert features["AMT_INCOME_TOTAL"].dtype == "float64"
    assert np.isnan(features.loc[0, "AMT_INCOME_TOTAL"])


def test_normalizes_categorical_pd_na(valid_frame):
    frame = valid_frame.copy()
    frame.loc[0, "CODE_GENDER"] = pd.NA

    features, _ = prepare_home_credit_features(frame, CONTRACT)

    assert features["CODE_GENDER"].dtype == "object"
    assert pd.isna(features.loc[0, "CODE_GENDER"])


def test_coerces_invalid_numeric_values_to_nan(valid_frame):
    frame = valid_frame.copy()
    frame["AMT_INCOME_TOTAL"] = frame["AMT_INCOME_TOTAL"].astype("object")
    frame.loc[0, "AMT_INCOME_TOTAL"] = "invalid"

    features, _ = prepare_home_credit_features(frame, CONTRACT)

    assert np.isnan(features.loc[0, "AMT_INCOME_TOTAL"])


def test_rejects_non_dataframe():
    with pytest.raises(TypeError):
        prepare_home_credit_features([1, 2], CONTRACT)


def test_rejects_empty_input():
    frame = pd.DataFrame(columns=CONTRACT["feature_order"])

    with pytest.raises(ValueError, match="no rows"):
        prepare_home_credit_features(frame, CONTRACT)


def test_rejects_missing_feature(valid_frame):
    frame = valid_frame.drop(columns=["CODE_GENDER"])

    with pytest.raises(ValueError, match="Missing"):
        prepare_home_credit_features(frame, CONTRACT)


def test_rejects_unexpected_column(valid_frame):
    frame = valid_frame.assign(UNEXPECTED=1)

    with pytest.raises(ValueError, match="Unexpected"):
        prepare_home_credit_features(frame, CONTRACT)


def test_rejects_duplicate_columns(valid_frame):
    frame = pd.concat(
        [valid_frame, valid_frame[["CNT_CHILDREN"]]],
        axis=1,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        prepare_home_credit_features(frame, CONTRACT)


def test_rejects_overlapping_feature_groups(valid_frame):
    contract = {
        **CONTRACT,
        "categorical_features": ["CODE_GENDER", "CNT_CHILDREN"],
    }

    with pytest.raises(ValueError, match="overlap"):
        prepare_home_credit_features(valid_frame, contract)


def test_rejects_incomplete_feature_groups(valid_frame):
    contract = {
        **CONTRACT,
        "numeric_features": ["AMT_INCOME_TOTAL"],
    }

    with pytest.raises(ValueError, match="cover"):
        prepare_home_credit_features(valid_frame, contract)
