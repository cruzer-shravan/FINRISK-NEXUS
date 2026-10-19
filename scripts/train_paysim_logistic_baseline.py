import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "modeling" / "paysim"
MODEL_DIR = ROOT / "models" / "paysim"
REPORT_DIR = ROOT / "reports" / "paysim"

TARGET = "isFraud"

NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
]

CATEGORICAL_FEATURES = ["type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def main():
    train_path = DATA_DIR / "train.parquet"
    validation_path = DATA_DIR / "validation.parquet"

    for path in [train_path, validation_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required modeling dataset not found: {path}")

    # Deliberately do not load the test dataset during model selection.
    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)

    required_columns = FEATURES + [TARGET]

    for name, df in [
        ("train", train),
        ("validation", validation),
    ]:
        missing = sorted(set(required_columns) - set(df.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

        if df[required_columns].isna().any().any():
            raise ValueError(f"Missing values found in {name} dataset.")

        if not df[TARGET].isin([0, 1]).all():
            raise ValueError(f"Unexpected target values in {name} dataset.")

        if df[TARGET].nunique() != 2:
            raise ValueError(f"{name} must contain both target classes.")

    X_train = train[FEATURES]
    y_train = train[TARGET].astype(int)

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET].astype(int)

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    print("=== TRAINING LOGISTIC REGRESSION BASELINE ===")
    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")
    print(f"Validation fraud cases: {int(y_validation.sum()):,}")
    print(f"Features: {FEATURES}")

    model.fit(X_train, y_train)

    # Validation predictions are used for model assessment
    # and decision-threshold selection.
    probabilities = model.predict_proba(X_validation)[:, 1]

    pr_auc = average_precision_score(y_validation, probabilities)
    roc_auc = roc_auc_score(y_validation, probabilities)

    default_metrics = calculate_metrics(y_validation, probabilities, 0.5)

    # Select the threshold that maximizes validation F1.
    # The test set is not used for this decision.
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_validation, probabilities)

    if len(thresholds) == 0:
        raise ValueError("Unable to determine thresholds.")

    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

    tuned_metrics = calculate_metrics(y_validation, probabilities, best_threshold)

    results = {
        "model": "LogisticRegression",
        "class_weight": "balanced",
        "features": FEATURES,
        "target": TARGET,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "training_fraud_cases": int(y_train.sum()),
        "validation_fraud_cases": int(y_validation.sum()),
        "validation_pr_auc": float(pr_auc),
        "validation_roc_auc": float(roc_auc),
        "validation_default_threshold": default_metrics,
        "validation_f1_selected_threshold": tuned_metrics,
        "threshold_selection": ("Maximum F1 on validation set; test set not used"),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "logistic_regression_baseline.joblib"
    metrics_path = REPORT_DIR / "logistic_baseline_metrics.json"

    joblib.dump(model, model_path)

    metrics_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n=== VALIDATION RESULTS ===")
    print(f"PR-AUC (Average Precision): {pr_auc:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")

    print("\n--- Default threshold: 0.50 ---")
    print(json.dumps(default_metrics, indent=2))

    print("\n--- F1-selected validation threshold ---")
    print(json.dumps(tuned_metrics, indent=2))

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("\nBaseline training completed successfully.")


if __name__ == "__main__":
    main()
