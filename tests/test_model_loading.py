import json

import joblib
import pandas as pd
import pytest
import src.inference.model_loader as loader
from sklearn.dummy import DummyClassifier


def make_model():
    model = DummyClassifier(strategy="prior")
    model.fit(
        pd.DataFrame(
            {
                "feature_a": [0.0, 1.0],
                "feature_b": [1.0, 0.0],
            }
        ),
        [0, 1],
    )
    return model


def test_calculate_sha256(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"test-artifact")

    digest = loader.calculate_sha256(artifact)

    assert len(digest) == 64
    assert digest == loader.calculate_sha256(artifact)


def test_load_contract_validates_required_fields(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "model_name": "test",
                "model_artifact": "model.joblib",
                "model_sha256": "abc",
                "feature_count": 2,
                "feature_order": ["a", "b"],
                "decision_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )

    contract = loader.load_contract(path)

    assert contract["feature_count"] == 2


def test_load_contract_rejects_missing_key(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({"model_name": "test"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        loader.load_contract(path)


def test_load_contract_rejects_feature_count_mismatch(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "model_name": "test",
                "model_artifact": "model.joblib",
                "model_sha256": "abc",
                "feature_count": 3,
                "feature_order": ["a", "b"],
                "decision_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feature count"):
        loader.load_contract(path)


def test_load_contract_rejects_duplicate_features(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "model_name": "test",
                "model_artifact": "model.joblib",
                "model_sha256": "abc",
                "feature_count": 2,
                "feature_order": ["a", "a"],
                "decision_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        loader.load_contract(path)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, "0.5"])
def test_load_contract_rejects_invalid_threshold(tmp_path, threshold):
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "model_name": "test",
                "model_artifact": "model.joblib",
                "model_sha256": "abc",
                "feature_count": 1,
                "feature_order": ["a"],
                "decision_threshold": threshold,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="threshold"):
        loader.load_contract(path)


def test_load_model_accepts_matching_artifact(tmp_path, monkeypatch):
    artifact = tmp_path / "model.joblib"
    joblib.dump(make_model(), artifact)

    monkeypatch.setattr(loader, "PROJECT_ROOT", tmp_path)

    contract = {
        "model_artifact": "model.joblib",
        "model_sha256": loader.calculate_sha256(artifact),
        "feature_count": 2,
        "feature_order": ["feature_a", "feature_b"],
    }

    loaded = loader.load_model(contract)

    assert list(loaded.classes_) == [0, 1]
    assert loaded.n_features_in_ == 2


def test_load_model_rejects_checksum_mismatch(tmp_path, monkeypatch):
    artifact = tmp_path / "model.joblib"
    joblib.dump(make_model(), artifact)

    monkeypatch.setattr(loader, "PROJECT_ROOT", tmp_path)

    contract = {
        "model_artifact": "model.joblib",
        "model_sha256": "0" * 64,
        "feature_count": 2,
        "feature_order": ["feature_a", "feature_b"],
    }

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        loader.load_model(contract)


def test_load_model_rejects_feature_order_mismatch(tmp_path, monkeypatch):
    artifact = tmp_path / "model.joblib"
    joblib.dump(make_model(), artifact)

    monkeypatch.setattr(loader, "PROJECT_ROOT", tmp_path)

    contract = {
        "model_artifact": "model.joblib",
        "model_sha256": loader.calculate_sha256(artifact),
        "feature_count": 2,
        "feature_order": ["feature_b", "feature_a"],
    }

    with pytest.raises(RuntimeError, match="feature order"):
        loader.load_model(contract)
