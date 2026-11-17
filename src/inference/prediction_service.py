"""Shared prediction interface for FINRISK NEXUS models."""

import numpy as np
import pandas as pd
from src.inference.home_credit_adapter import prepare_home_credit_features
from src.inference.model_loader import load_model_bundle
from src.inference.paysim_adapter import prepare_paysim_features

CONTRACT_PATHS = {
    "paysim": "configs/model_contracts/paysim_feature_contract.json",
    "home_credit": "configs/model_contracts/home_credit_feature_contract.json",
}


class PredictionService:
    def __init__(self, dataset: str):
        if dataset not in CONTRACT_PATHS:
            raise ValueError(
                f"Unsupported dataset '{dataset}'. Choose from {sorted(CONTRACT_PATHS)}."
            )

        self.dataset = dataset
        self.contract, self.model = load_model_bundle(CONTRACT_PATHS[dataset])

        self.positive_class_index = list(self.model.classes_).index(1)

    def prepare(self, frame: pd.DataFrame):
        if self.dataset == "paysim":
            features = prepare_paysim_features(frame, self.contract)
            identifiers = None
        else:
            features, identifiers = prepare_home_credit_features(frame, self.contract)

        return features, identifiers

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        features, identifiers = self.prepare(frame)

        probabilities = self.model.predict_proba(features)[:, self.positive_class_index]

        if not np.isfinite(probabilities).all():
            raise RuntimeError("Model returned non-finite probabilities.")

        threshold = float(self.contract["decision_threshold"])
        predictions = (probabilities >= threshold).astype("int8")

        result = pd.DataFrame(
            {
                "risk_probability": probabilities,
                "predicted_class": predictions,
                "decision_threshold": threshold,
                "model_name": self.contract["model_name"],
            },
            index=frame.index,
        )

        if identifiers is not None:
            result.insert(0, "SK_ID_CURR", identifiers)

        return result
