import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "modeling" / "paysim"
MODEL_PATH = ROOT / "models" / "paysim" / "logistic_regression_baseline.joblib"
METRICS_PATH = ROOT / "reports" / "paysim" / "logistic_baseline_metrics.json"

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "type",
]

TARGET = "isFraud"


def main():
    validation = pd.read_parquet(DATA_DIR / "validation.parquet")
    model = joblib.load(MODEL_PATH)

    X = validation[FEATURES]
    y = validation[TARGET].astype(int).to_numpy()
    probabilities = model.predict_proba(X)[:, 1]

    print("=== PROBABILITY DISTRIBUTION ===")
    print(pd.Series(probabilities).describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]).to_string())

    print("\n=== VALIDATION THRESHOLD COMPARISON ===")
    print(f"PR-AUC: {average_precision_score(y, probabilities):.6f}")
    print(f"Fraud prevalence: {y.mean():.6%}")

    thresholds = [0.50, 0.90, 0.99, 0.999, 0.9999]

    metrics = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        metrics.append(
            {
                "threshold": threshold,
                "flagged": int(predictions.sum()),
                "true_positives": int(((predictions == 1) & (y == 1)).sum()),
                "false_positives": int(((predictions == 1) & (y == 0)).sum()),
                "false_negatives": int(((predictions == 0) & (y == 1)).sum()),
                "precision": precision_score(y, predictions, zero_division=0),
                "recall": recall_score(y, predictions, zero_division=0),
                "f1": f1_score(y, predictions, zero_division=0),
            }
        )

    print(pd.DataFrame(metrics).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    with METRICS_PATH.open(encoding="utf-8") as f:
        saved = json.load(f)

    selected = saved["validation_f1_selected_threshold"]["threshold"]
    selected_predictions = (probabilities >= selected).astype(int)

    print("\n=== SELECTED THRESHOLD CHECK ===")
    print(f"Selected threshold: {selected:.12f}")
    print(f"Predicted probability maximum: {probabilities.max():.12f}")
    print(
        "Predicted positives at selected threshold:",
        int(selected_predictions.sum()),
    )
    print(
        "Predicted positives at threshold 0.5:",
        int((probabilities >= 0.5).sum()),
    )


if __name__ == "__main__":
    main()
