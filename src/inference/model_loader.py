"""Secure, contract-driven loading of frozen model artifacts."""

import hashlib
import json
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_contract(contract_path: str | Path) -> dict:
    path = Path(contract_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"Contract not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        contract = json.load(file)

    required = {
        "model_name",
        "model_artifact",
        "model_sha256",
        "feature_count",
        "feature_order",
        "decision_threshold",
    }

    missing = required - set(contract)
    if missing:
        raise ValueError(f"Contract missing required keys: {sorted(missing)}")

    if len(contract["feature_order"]) != contract["feature_count"]:
        raise ValueError("Contract feature count does not match feature order.")

    if len(set(contract["feature_order"])) != len(contract["feature_order"]):
        raise ValueError("Contract contains duplicate feature names.")

    threshold = contract["decision_threshold"]
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must be between 0 and 1.")

    return contract


def load_model(contract: dict):
    artifact_path = PROJECT_ROOT / contract["model_artifact"]

    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    actual_hash = calculate_sha256(artifact_path)
    expected_hash = contract["model_sha256"]

    if actual_hash.lower() != expected_hash.lower():
        raise RuntimeError(
            f"Model artifact checksum mismatch. Expected {expected_hash}, got {actual_hash}."
        )

    model = joblib.load(artifact_path)

    model_features = list(getattr(model, "feature_names_in_", []))
    contract_features = contract["feature_order"]

    if model_features and model_features != contract_features:
        raise RuntimeError("Loaded model feature order does not match the contract.")

    if getattr(model, "n_features_in_", None) != contract["feature_count"]:
        raise RuntimeError("Loaded model feature count does not match contract.")

    if 1 not in list(getattr(model, "classes_", [])):
        raise RuntimeError("Loaded model does not expose positive class 1.")

    return model


def load_model_bundle(contract_path: str | Path) -> tuple[dict, object]:
    contract = load_contract(contract_path)
    model = load_model(contract)
    return contract, model
