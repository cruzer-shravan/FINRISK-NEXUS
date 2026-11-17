"""Input validation and feature selection for the frozen PaySim model."""

import numpy as np
import pandas as pd

ALLOWED_TRANSACTION_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
}


def prepare_paysim_features(
    frame: pd.DataFrame,
    contract: dict,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("PaySim input must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError("PaySim input contains no rows.")

    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate input columns: {duplicates}")

    expected = contract["feature_order"]
    missing = sorted(set(expected) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(expected))

    if missing:
        raise ValueError(f"Missing PaySim features: {missing}")

    if unexpected:
        raise ValueError(f"Unexpected PaySim columns: {unexpected}")

    result = frame.loc[:, expected].copy()

    if result.isna().any().any():
        raise ValueError("PaySim input contains missing values.")

    numeric_features = contract["numeric_features"]

    for column in numeric_features:
        values = pd.to_numeric(result[column], errors="coerce")

        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{column} must contain finite numeric values.")

        result[column] = values.astype("float64")

    nonnegative = [
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
    ]

    for column in nonnegative:
        if (result[column] < 0).any():
            raise ValueError(f"{column} cannot contain negative values.")

    expected_log = np.log1p(result["amount"].to_numpy())
    actual_log = result["log_amount"].to_numpy()

    if not np.allclose(
        actual_log,
        expected_log,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError("log_amount is inconsistent with log1p(amount).")

    transaction_types = set(result["type"].dropna().unique())
    invalid_types = transaction_types - ALLOWED_TRANSACTION_TYPES

    if invalid_types:
        raise ValueError(f"Unsupported PaySim transaction types: {sorted(invalid_types)}")

    result["type"] = result["type"].astype("object")

    return result
