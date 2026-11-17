import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42
N_SPLITS = 3

TRAIN_PATH = Path("data/modeling/home_credit/application_only/train.parquet")
VALIDATION_PATH = Path("data/modeling/home_credit/application_only/validation.parquet")

MODEL_DIR = Path("models/home_credit/application_only")
REPORT_DIR = Path("reports/home_credit/application_only")

CANDIDATES = {
    "xgboost_tuned": MODEL_DIR / "xgboost_tuned.joblib",
    "lightgbm_tuned": MODEL_DIR / "lightgbm_tuned.joblib",
}

METHODS = ["sigmoid", "isotonic"]

OUTPUT_MODELS = [
    MODEL_DIR / f"calibrated_{model_name}_{method}.joblib"
    for model_name in CANDIDATES
    for method in METHODS
]

OUTPUT_METRICS = [
    REPORT_DIR / f"calibrated_{model_name}_{method}_metrics.json"
    for model_name in CANDIDATES
    for method in METHODS
]

SUMMARY_PATH = REPORT_DIR / "calibration_comparison.csv"
RELIABILITY_PATH = REPORT_DIR / "calibration_reliability.csv"
CHART_PATH = REPORT_DIR / "calibration_comparison.png"

OUTPUTS = OUTPUT_MODELS + OUTPUT_METRICS + [SUMMARY_PATH, RELIABILITY_PATH, CHART_PATH]

for path in OUTPUTS:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}. Inspect before overwriting.")

for path in CANDIDATES.values():
    if not path.exists():
        raise FileNotFoundError(f"Required model not found: {path}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Home Credit training and validation partitions...")

train_df = pd.read_parquet(TRAIN_PATH)
valid_df = pd.read_parquet(VALIDATION_PATH)

TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"

X_train = train_df.drop(columns=[TARGET, ID_COLUMN]).copy()
y_train = train_df[TARGET].astype("int8")

X_valid = valid_df.drop(columns=[TARGET, ID_COLUMN]).copy()
y_valid = valid_df[TARGET].astype("int8")

# Convert Pandas nullable values into NumPy-compatible missing values.
numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()

categorical_columns = X_train.select_dtypes(exclude=["number"]).columns.tolist()

for column in numeric_columns:
    X_train[column] = pd.to_numeric(X_train[column], errors="coerce").astype("float64")

    X_valid[column] = pd.to_numeric(X_valid[column], errors="coerce").astype("float64")

for column in categorical_columns:
    X_train[column] = X_train[column].astype("object")
    X_train[column] = X_train[column].where(pd.notna(X_train[column]), np.nan)

    X_valid[column] = X_valid[column].astype("object")
    X_valid[column] = X_valid[column].where(pd.notna(X_valid[column]), np.nan)

print(f"Training rows: {len(X_train):,}")
print(f"Validation rows: {len(X_valid):,}")
print(f"Training defaults: {int(y_train.sum()):,}")
print(f"Validation defaults: {int(y_valid.sum()):,}")
print(f"Cross-validation folds: {N_SPLITS}")

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

results = []
reliability_rows = []

for model_name, model_path in CANDIDATES.items():
    print("\n" + "=" * 70)
    print(f"CALIBRATION: {model_name.upper()}")
    print("=" * 70)

    base_model = joblib.load(model_path)

    if not hasattr(base_model, "predict_proba"):
        raise TypeError(f"{model_name} does not support predict_proba.")

    print("Evaluating original model on validation data...")

    original_probabilities = base_model.predict_proba(X_valid)[:, 1]

    original_metrics = {
        "model": model_name,
        "method": "uncalibrated",
        "brier_score": float(brier_score_loss(y_valid, original_probabilities)),
        "roc_auc": float(roc_auc_score(y_valid, original_probabilities)),
        "average_precision": float(average_precision_score(y_valid, original_probabilities)),
        "log_loss": float(log_loss(y_valid, original_probabilities)),
    }

    results.append(original_metrics)

    original_fraction, original_mean = calibration_curve(
        y_valid,
        original_probabilities,
        n_bins=10,
        strategy="quantile",
    )

    for bin_number, (mean_predicted, observed) in enumerate(
        zip(original_mean, original_fraction, strict=True), start=1
    ):
        reliability_rows.append(
            {
                "model": model_name,
                "method": "uncalibrated",
                "bin": bin_number,
                "mean_predicted_probability": float(mean_predicted),
                "observed_default_rate": float(observed),
            }
        )

    for method in METHODS:
        print(f"\nFitting {method.upper()} calibration...")

        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv=cv,
            ensemble=False,
            n_jobs=1,
        )

        calibrated_model.fit(X_train, y_train)

        probabilities = calibrated_model.predict_proba(X_valid)[:, 1]

        metrics = {
            "model": model_name,
            "method": method,
            "calibration_cv_folds": N_SPLITS,
            "calibration_training_rows": int(len(X_train)),
            "validation_rows": int(len(X_valid)),
            "brier_score": float(brier_score_loss(y_valid, probabilities)),
            "roc_auc": float(roc_auc_score(y_valid, probabilities)),
            "average_precision": float(average_precision_score(y_valid, probabilities)),
            "log_loss": float(log_loss(y_valid, probabilities)),
        }

        model_output = MODEL_DIR / f"calibrated_{model_name}_{method}.joblib"

        metrics_output = REPORT_DIR / f"calibrated_{model_name}_{method}_metrics.json"

        joblib.dump(calibrated_model, model_output)

        with metrics_output.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        results.append(metrics)

        fraction, mean = calibration_curve(
            y_valid,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )

        for bin_number, (mean_predicted, observed) in enumerate(
            zip(mean, fraction, strict=True), start=1
        ):
            reliability_rows.append(
                {
                    "model": model_name,
                    "method": method,
                    "bin": bin_number,
                    "mean_predicted_probability": float(mean_predicted),
                    "observed_default_rate": float(observed),
                }
            )

        print(f"Brier score:       {metrics['brier_score']:.6f}")
        print(f"ROC-AUC:           {metrics['roc_auc']:.6f}")
        print(f"Average Precision: {metrics['average_precision']:.6f}")
        print(f"Log Loss:          {metrics['log_loss']:.6f}")
        print(f"Saved model:       {model_output}")

comparison = pd.DataFrame(results)
comparison.to_csv(SUMMARY_PATH, index=False)

reliability_df = pd.DataFrame(reliability_rows)
reliability_df.to_csv(RELIABILITY_PATH, index=False)

# Plot reliability curves for original and calibrated models.
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, model_name in zip(axes, CANDIDATES, strict=True):
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    model_data = reliability_df[reliability_df["model"] == model_name]

    for method in ["uncalibrated", "sigmoid", "isotonic"]:
        method_data = model_data[model_data["method"] == method].sort_values("bin")

        ax.plot(
            method_data["mean_predicted_probability"],
            method_data["observed_default_rate"],
            marker="o",
            label=method.capitalize(),
        )

    ax.set_title(model_name.replace("_", " ").title())
    ax.set_xlabel("Mean predicted default probability")
    ax.set_ylabel("Observed default rate")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle("Home Credit — Boosting Model Calibration")
fig.tight_layout()
fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

print("\n" + "=" * 70)
print("CALIBRATION COMPARISON")
print("=" * 70)

print(
    comparison[
        [
            "model",
            "method",
            "brier_score",
            "roc_auc",
            "average_precision",
            "log_loss",
        ]
    ].to_string(index=False)
)

print(f"\nSaved comparison: {SUMMARY_PATH}")
print(f"Saved reliability data: {RELIABILITY_PATH}")
print(f"Saved chart: {CHART_PATH}")

print("\nCalibration experiment completed.")
print("The test partition was not loaded or evaluated.")
