import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "modeling" / "paysim"
MODEL_DIR = ROOT / "models" / "paysim" / "tree_fraud_v1"
REPORT_DIR = ROOT / "reports" / "paysim" / "tree_fraud_v1"

TARGET = "isFraud"
NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
]
CATEGORICAL_FEATURES = ["type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


def calculate_threshold_metrics(y_true, probabilities, threshold):
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
        "flagged_transactions": int(tp + fp),
    }


def select_f1_threshold(y_true, probabilities):
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        raise ValueError("No candidate thresholds were produced.")

    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index])


def make_preprocessor():
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def validate_partition(name, df):
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError(f"{name} has missing values in required columns.")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(f"{name} has unexpected target values.")
    if df[TARGET].nunique() != 2:
        raise ValueError(f"{name} must contain both target classes.")
    if not set(df["type"].unique()).issubset(
        {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
    ):
        raise ValueError(f"{name} contains unexpected transaction types.")


def main():
    train_path = DATA_DIR / "train.parquet"
    validation_path = DATA_DIR / "validation.parquet"

    for path in (train_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found: {path}")

    # Intentionally do not load test.parquet during model selection.
    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)

    validate_partition("train", train)
    validate_partition("validation", validation)

    X_train = train[FEATURES]
    y_train = train[TARGET].astype("int8")
    X_validation = validation[FEATURES]
    y_validation = validation[TARGET].astype("int8")

    candidates = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=100,
            max_samples=0.5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    comparison = []

    print("=== PAYSIM TREE MODEL EXPERIMENT ===")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")
    print(f"Validation fraud cases: {int(y_validation.sum()):,}")
    print(f"Features: {FEATURES}")
    print("Test partition will not be loaded.")

    for name, estimator in candidates.items():
        print(f"\n=== TRAINING {name.upper()} ===")
        start = time.perf_counter()

        model = Pipeline(
            [
                ("preprocessing", make_preprocessor()),
                ("classifier", estimator),
            ]
        )
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_validation)[:, 1]
        elapsed = time.perf_counter() - start

        threshold = select_f1_threshold(y_validation, probabilities)
        tuned_metrics = calculate_threshold_metrics(y_validation, probabilities, threshold)
        default_metrics = calculate_threshold_metrics(y_validation, probabilities, 0.5)

        results = {
            "experiment_id": "paysim-tree-fraud-v1",
            "model": name,
            "target": TARGET,
            "features": FEATURES,
            "random_state": RANDOM_STATE,
            "scikit_learn_version": sklearn.__version__,
            "training_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "training_fraud_cases": int(y_train.sum()),
            "validation_fraud_cases": int(y_validation.sum()),
            "training_fraud_rate": float(y_train.mean()),
            "validation_fraud_rate": float(y_validation.mean()),
            "validation_average_precision": float(
                average_precision_score(y_validation, probabilities)
            ),
            "validation_roc_auc": float(roc_auc_score(y_validation, probabilities)),
            "validation_default_threshold": default_metrics,
            "validation_f1_selected_threshold": tuned_metrics,
            "threshold_selection": ("Maximum F1 on validation predictions; test not loaded"),
            "training_and_validation_seconds": round(elapsed, 2),
            "model_parameters": estimator.get_params(),
        }

        model_path = MODEL_DIR / f"{name}.joblib"
        report_path = REPORT_DIR / f"{name}_metrics.json"

        joblib.dump(model, model_path)
        report_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

        comparison.append(
            {
                "model": name,
                "validation_average_precision": results["validation_average_precision"],
                "validation_roc_auc": results["validation_roc_auc"],
                "validation_f1": tuned_metrics["f1"],
                "validation_precision": tuned_metrics["precision"],
                "validation_recall": tuned_metrics["recall"],
                "selected_threshold": threshold,
                "flagged_transactions": tuned_metrics["flagged_transactions"],
                "elapsed_seconds": round(elapsed, 2),
            }
        )

        print(f"Average Precision: {results['validation_average_precision']:.6f}")
        print(f"ROC-AUC: {results['validation_roc_auc']:.6f}")
        print("F1-selected validation metrics:")
        print(json.dumps(tuned_metrics, indent=2))
        print(f"Saved model: {model_path}")
        print(f"Saved metrics: {report_path}")

    comparison_df = pd.DataFrame(comparison).sort_values(
        "validation_average_precision", ascending=False
    )
    comparison_path = REPORT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n=== VALIDATION MODEL COMPARISON ===")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved comparison: {comparison_path}")
    print("\nTree-model validation runs completed.")
    print("No test-set evaluation has been performed.")


if __name__ == "__main__":
    main()
