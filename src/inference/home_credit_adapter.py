"""Input normalization for the frozen Home Credit model."""

import numpy as np
import pandas as pd

ALLOWED_METADATA_COLUMNS = {"SK_ID_CURR"}


def prepare_home_credit_features(
    frame: pd.DataFrame,
    contract: dict,
) -> tuple[pd.DataFrame, pd.Series | None]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Home Credit input must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError("Home Credit input contains no rows.")

    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate input columns: {duplicates}")

    expected = contract["feature_order"]
    allowed = set(expected) | ALLOWED_METADATA_COLUMNS

    missing = sorted(set(expected) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - allowed)

    if missing:
        raise ValueError(f"Missing Home Credit features: {missing}")

    if unexpected:
        raise ValueError(f"Unexpected Home Credit columns: {unexpected}")

    identifiers = None

    if "SK_ID_CURR" in frame.columns:
        identifiers = frame["SK_ID_CURR"].copy()

    # Restore exact model feature order.
    result = frame.loc[:, expected].copy()

    numeric_features = contract["numeric_features"]
    categorical_features = contract["categorical_features"]

    if set(numeric_features) & set(categorical_features):
        raise ValueError("Numeric and categorical feature lists overlap.")

    if set(numeric_features) | set(categorical_features) != set(expected):
        raise ValueError("Numeric and categorical features do not cover the model schema.")

    # Match the training/calibration input preparation:
    # numeric values -> float64, invalid values -> np.nan.
    for column in numeric_features:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).astype("float64")

    # Categorical values -> object, all missing sentinels -> np.nan.
    for column in categorical_features:
        result[column] = result[column].astype("object").where(result[column].notna(), np.nan)

    return result, identifiers
