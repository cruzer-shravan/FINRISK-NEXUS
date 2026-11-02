"""
Evaluate the saved PaySim tree-model candidate on the held-out test partition.

This script does not fit/retrain a model and does not select a threshold from
test labels. It reuses the threshold selected on the validation partition.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "modeling" / "paysim"
MODEL_DIR = ROOT / "models" / "paysim" / "tree_fraud_v1"
REPORT_DIR = ROOT / "reports" / "paysim" / "tree_fraud_v1"

MODEL_NAME = "decision_tree"
TARGET = "isFraud"
FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "type",
]


def load_threshold(metrics_path: Path) -> float:
    """Read the threshold selected using validation data only."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Validation metrics not found: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold_block = metrics.get("validation_f1_selected_threshold")
    if not isinstance(threshold_block, dict) or "threshold" not in threshold_block:
        raise ValueError(f"Expected 'validation_f1_selected_threshold.threshold' in {metrics_path}")

    threshold = float(threshold_block["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid saved threshold: {threshold}")

    return threshold


def main() -> None:
    model_path = MODEL_DIR / f"{MODEL_NAME}.joblib"
    validation_metrics_path = REPORT_DIR / f"{MODEL_NAME}_metrics.json"
    test_path = DATA_DIR / "test.parquet"

    for path in (model_path, validation_metrics_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    # Load the already-trained model and validation-selected threshold.
    model = joblib.load(model_path)
    threshold = load_threshold(validation_metrics_path)

    # This is the only partition read by this evaluation script.
    test = pd.read_parquet(test_path)

    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(test.columns))
    if missing:
        raise ValueError(f"Test partition is missing columns: {missing}")
    if test[required].isna().any().any():
        raise ValueError("Test partition contains missing required values.")
    if not test[TARGET].isin([0, 1]).all():
        raise ValueError("Test target contains values other than 0 and 1.")
    if test[TARGET].nunique() != 2:
        raise ValueError("Test partition must contain both target classes.")

    X_test = test[FEATURES]
    y_test = test[TARGET].astype("int8")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype("int8")

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    results = {
        "experiment_id": "paysim-tree-fraud-v1",
        "evaluation_type": "locked_holdout_test_evaluation",
        "model": MODEL_NAME,
        "target": TARGET,
        "features": FEATURES,
        "threshold": threshold,
        "threshold_source": "validation_f1_selected_threshold",
        "threshold_tuned_on_test": False,
        "model_retrained_during_evaluation": False,
        "scikit_learn_version": sklearn.__version__,
        "evaluation_timestamp_utc": datetime.now(UTC).isoformat(),
        "test_rows": int(len(test)),
        "test_fraud_cases": int(y_test.sum()),
        "test_fraud_rate": float(y_test.mean()),
        "test_min_step": int(test["step"].min()) if "step" in test.columns else None,
        "test_max_step": int(test["step"].max()) if "step" in test.columns else None,
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "flagged_transactions": int(tp + fp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fraud_capture_rate": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "notes": [
            "Threshold was selected on validation data and frozen before test evaluation.",
            "Test metrics are for a later chronological period "
            "with a different observed fraud prevalence.",
            "Do not use these test results to tune the model or "
            "continue treating this same partition as an untouched holdout.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_output = REPORT_DIR / f"{MODEL_NAME}_test_metrics.json"
    confusion_output = REPORT_DIR / f"{MODEL_NAME}_test_confusion_matrix.csv"

    metrics_output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    confusion_df = pd.DataFrame(
        [
            {"actual": "legitimate", "predicted_legitimate": tn, "predicted_fraud": fp},
            {"actual": "fraud", "predicted_legitimate": fn, "predicted_fraud": tp},
        ]
    )
    confusion_df.to_csv(confusion_output, index=False)

    print("=== PAYSIM DECISION TREE: LOCKED TEST EVALUATION ===")
    print(f"Model: {model_path}")
    print(f"Threshold (selected on validation): {threshold:.6f}")
    print(f"Test rows: {len(test):,}")
    print(f"Test fraud cases: {int(y_test.sum()):,}")
    print(f"Test fraud rate: {y_test.mean():.6%}")
    print(f"Average Precision: {results['average_precision']:.6f}")
    print(f"ROC-AUC: {results['roc_auc']:.6f}")
    print(f"Precision: {results['precision']:.6f}")
    print(f"Recall: {results['recall']:.6f}")
    print(f"F1: {results['f1']:.6f}")
    print("\nConfusion matrix counts:")
    print(f"  True positives:  {tp:,}")
    print(f"  False positives: {fp:,}")
    print(f"  True negatives:  {tn:,}")
    print(f"  False negatives: {fn:,}")
    print(f"\nSaved metrics: {metrics_output}")
    print(f"Saved confusion matrix: {confusion_output}")
    print("\nTest evaluation completed. No training or threshold tuning was performed.")


if __name__ == "__main__":
    main()
