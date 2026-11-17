from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import xgboost

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "reports/home_credit/model_selection_locks/"
    "20261107_xgboost_sigmoid/model/"
    "calibrated_xgboost_tuned_sigmoid.joblib"
)

TEST_PATH = ROOT / "data/modeling/home_credit/application_only/test.parquet"

OUTPUT_DIR = ROOT / "reports/home_credit/test_evaluation/20261107_xgboost_sigmoid"

EXPECTED_MODEL_SHA256 = "21B24D48DD54A03080F30D3421D807D74AE04FD7F734C34DA82D3E59E33D644B"

THRESHOLD = 0.17248130501689904
FP_COST = 1
FN_COST = 5

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"
RANDOM_STATE = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Mirror the feature preparation used in validation business-cost analysis."""
    X = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN]).copy()

    numeric_columns = X.select_dtypes(include=["number"]).columns
    categorical_columns = X.columns.difference(numeric_columns)

    for column in numeric_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce").astype("float64")

    for column in categorical_columns:
        X[column] = X[column].astype("object")
        X[column] = X[column].where(X[column].notna(), np.nan)

    return X


def main() -> None:
    print("=" * 72)
    print("FINRISK NEXUS — HOME CREDIT LOCKED MODEL TEST EVALUATION")
    print("=" * 72)

    # ---------------------------------------------------------------
    # 1. Pre-evaluation integrity checks
    # ---------------------------------------------------------------
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Locked model not found: {MODEL_PATH}")

    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"Test partition not found: {TEST_PATH}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}\n"
            "No files were overwritten. Use a new, reviewed evaluation ID "
            "only if a separate evaluation is explicitly authorized."
        )

    model_hash_before = sha256_file(MODEL_PATH).upper()
    if model_hash_before != EXPECTED_MODEL_SHA256.upper():
        raise RuntimeError(
            "LOCKED MODEL HASH MISMATCH. Evaluation aborted.\n"
            f"Expected: {EXPECTED_MODEL_SHA256}\n"
            f"Actual:   {model_hash_before}"
        )

    test_hash_before = sha256_file(TEST_PATH)

    print("Locked model hash: VERIFIED")
    print("Test partition hash: RECORDED")
    print("Decision threshold:", THRESHOLD)
    print("Cost assumptions: FP =", FP_COST, "| FN =", FN_COST)

    # ---------------------------------------------------------------
    # 2. Load the locked model and test partition
    # ---------------------------------------------------------------
    print("\nLoading locked model...")
    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise TypeError("Locked model does not support predict_proba.")

    model_classes = list(model.classes_)
    if 1 not in model_classes or 0 not in model_classes:
        raise ValueError(f"Unexpected model classes: {model_classes}")

    positive_class_index = model_classes.index(1)

    print("Model type:", type(model).__name__)
    print("Model classes:", model_classes)

    print("\nLoading test partition for the first independent evaluation...")
    test_frame = pd.read_parquet(TEST_PATH)

    if TARGET_COLUMN not in test_frame.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    if ID_COLUMN not in test_frame.columns:
        raise ValueError(f"Missing identifier column: {ID_COLUMN}")

    if test_frame[TARGET_COLUMN].isna().any():
        raise ValueError("Missing target values detected. Evaluation aborted.")

    if test_frame[ID_COLUMN].isna().any():
        raise ValueError("Missing customer IDs detected. Evaluation aborted.")

    if not test_frame[ID_COLUMN].is_unique:
        raise ValueError("Duplicate customer IDs detected. Evaluation aborted.")

    observed_targets = set(test_frame[TARGET_COLUMN].unique().tolist())
    if observed_targets != {0, 1}:
        raise ValueError(f"Unexpected target values: {observed_targets}")

    y_test = test_frame[TARGET_COLUMN].astype("int8")
    X_test = prepare_features(test_frame)

    del test_frame

    print("Test rows:", len(y_test))
    print("Feature count:", X_test.shape[1])
    print("Actual defaults:", int(y_test.sum()))
    print("Actual non-defaults:", int((y_test == 0).sum()))

    # ---------------------------------------------------------------
    # 3. Generate probabilities and apply the pre-locked threshold
    # ---------------------------------------------------------------
    print("\nGenerating test probabilities...")
    probabilities_all = model.predict_proba(X_test)
    probabilities = np.asarray(
        probabilities_all[:, positive_class_index],
        dtype="float64",
    )

    if len(probabilities) != len(y_test):
        raise ValueError("Prediction count does not match test row count.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities detected.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities outside [0, 1] detected.")

    # The threshold is fixed from validation. It is never optimized here.
    predictions = (probabilities >= THRESHOLD).astype("int8")

    # ---------------------------------------------------------------
    # 4. Calculate independent test metrics
    # ---------------------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    total_business_cost = int(FP_COST * fp + FN_COST * fn)

    metrics = {
        "evaluation_name": "home_credit_locked_model_independent_test",
        "evaluation_timestamp_utc": datetime.now(UTC).isoformat(),
        "evaluation_status": "COMPLETED",
        "model_name": "Tuned XGBoost Sigmoid",
        "model_type": type(model).__name__,
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "model_sha256": model_hash_before,
        "test_dataset": str(TEST_PATH.relative_to(ROOT)),
        "test_dataset_sha256_before": test_hash_before,
        "test_rows": int(len(y_test)),
        "feature_count": int(X_test.shape[1]),
        "actual_default_cases": int(y_test.sum()),
        "actual_non_default_cases": int((y_test == 0).sum()),
        "actual_default_rate": float(y_test.mean()),
        "decision_threshold": THRESHOLD,
        "threshold_source": "pre-approved validation decision policy",
        "threshold_optimized_on_test": False,
        "false_positive_cost": FP_COST,
        "false_negative_cost": FN_COST,
        "business_cost_formula": "FP * 1 + FN * 5",
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "total_business_cost": total_business_cost,
        "business_cost_per_application": float(total_business_cost / len(y_test)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_test, predictions, beta=2, zero_division=0)),
        "predicted_positive_rate": float(predictions.mean()),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "test_threshold_optimization_performed": False,
        "model_retrained": False,
    }

    # ---------------------------------------------------------------
    # 5. Calibration diagnostics — descriptive only
    # ---------------------------------------------------------------
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    reliability = pd.DataFrame(
        {
            "mean_predicted_probability": mean_predicted_value,
            "observed_default_rate": fraction_of_positives,
        }
    )

    # Weighted expected calibration error using quantile bins.
    bin_ids = pd.qcut(
        probabilities,
        q=10,
        labels=False,
        duplicates="drop",
    )
    calibration_frame = pd.DataFrame(
        {
            "bin": bin_ids,
            "actual": y_test.to_numpy(),
            "probability": probabilities,
        }
    )

    ece = 0.0
    for _, group in calibration_frame.groupby("bin", observed=True):
        if len(group) == 0:
            continue
        ece += (len(group) / len(calibration_frame)) * abs(
            float(group["actual"].mean()) - float(group["probability"].mean())
        )

    metrics["calibration"] = {
        "method": "quantile bins",
        "requested_bins": 10,
        "actual_bins": int(len(reliability)),
        "expected_calibration_error": float(ece),
        "ece_definition": (
            "Weighted mean absolute difference between observed default "
            "rate and mean predicted probability across quantile bins."
        ),
    }

    # ---------------------------------------------------------------
    # 6. Verify input files remain unchanged and create outputs
    # ---------------------------------------------------------------
    model_hash_after = sha256_file(MODEL_PATH).upper()
    test_hash_after = sha256_file(TEST_PATH)

    if model_hash_after != model_hash_before:
        raise RuntimeError("Locked model file changed during evaluation.")

    if test_hash_after != test_hash_before:
        raise RuntimeError("Test partition file changed during evaluation.")

    metrics["test_dataset_sha256_after"] = test_hash_after
    metrics["model_sha256_after"] = model_hash_after
    metrics["input_integrity_check"] = "PASSED"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    with (OUTPUT_DIR / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, allow_nan=False)

    reliability.to_csv(
        OUTPUT_DIR / "calibration_reliability.csv",
        index=False,
    )

    pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual non-default", "Actual default"],
        columns=["Predicted non-default", "Predicted default"],
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    # Calibration plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        label="Locked model",
    )
    ax.set_xlabel("Mean predicted default probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Home Credit — Independent Test Calibration")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_calibration_curve.png", dpi=150)
    plt.close(fig)

    # Human-readable audit report
    report = f"""# Home Credit — Independent Test Evaluation

## Evaluation status

**{metrics["evaluation_status"]}**

This is the first independent evaluation of the provisionally approved,
locked model on the untouched test partition.

## Model and decision policy

- Model: Tuned XGBoost with sigmoid calibration
- Model artifact: `{MODEL_PATH.relative_to(ROOT)}`
- Model SHA256: `{model_hash_before}`
- Fixed decision threshold: `{THRESHOLD:.17f}`
- Threshold source: pre-approved validation decision policy
- False-positive cost: {FP_COST}
- False-negative cost: {FN_COST}
- Test threshold optimization: **Not performed**
- Model retraining: **Not performed**

## Test partition

- Rows: {len(y_test):,}
- Features: {X_test.shape[1]}
- Actual defaults: {int(y_test.sum()):,}
- Actual non-defaults: {int((y_test == 0).sum()):,}
- Actual default rate: {y_test.mean():.4%}
- Test dataset SHA256: `{test_hash_after}`

## Classification performance

| Metric | Test result |
|---|---:|
| ROC-AUC | {metrics["roc_auc"]:.6f} |
| Average Precision | {metrics["average_precision"]:.6f} |
| Brier Score | {metrics["brier_score"]:.6f} |
| Log Loss | {metrics["log_loss"]:.6f} |
| Accuracy | {metrics["accuracy"]:.6f} |
| Precision | {metrics["precision"]:.6f} |
| Recall | {metrics["recall"]:.6f} |
| F1 | {metrics["f1"]:.6f} |
| F2 | {metrics["f2"]:.6f} |

## Confusion matrix

| | Predicted non-default | Predicted default |
|---|---:|---:|
| Actual non-default | {tn:,} (TN) | {fp:,} (FP) |
| Actual default | {fn:,} (FN) | {tp:,} (TP) |

## Business-cost evaluation

Formula: `FP × 1 + FN × 5`

- False positives: {fp:,}
- False negatives: {fn:,}
- Total test business cost: **{total_business_cost:,}**
- Cost per application: {total_business_cost / len(y_test):.6f}

These cost weights are the illustrative assumptions approved for this
evaluation; they are not actual lender loss estimates.

## Calibration diagnostics

- Quantile bins requested: 10
- Actual bins: {len(reliability)}
- Expected Calibration Error: {ece:.6f}

See `calibration_reliability.csv` and `test_calibration_curve.png`.

## Integrity and evaluation controls

- Locked model hash verified before evaluation: PASS
- Locked model hash unchanged after evaluation: PASS
- Test partition hash unchanged after evaluation: PASS
- Threshold optimized on test: NO
- Model retrained: NO
- Customer-level prediction file generated: NO

## Interpretation restriction

Test results are an independent estimate of generalization performance.
They must not be used to tune the threshold or select a replacement model.
Any future model or policy change requires a new development and validation
cycle with a separate untouched test set.
"""

    (OUTPUT_DIR / "TEST_EVALUATION_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("INDEPENDENT TEST EVALUATION COMPLETED")
    print("=" * 72)
    print(f"ROC-AUC:          {metrics['roc_auc']:.6f}")
    print(f"Average Precision:{metrics['average_precision']:.6f}")
    print(f"Brier Score:      {metrics['brier_score']:.6f}")
    print(f"Precision:        {metrics['precision']:.6f}")
    print(f"Recall:           {metrics['recall']:.6f}")
    print(f"F1:               {metrics['f1']:.6f}")
    print(f"F2:               {metrics['f2']:.6f}")
    print(f"Threshold:        {THRESHOLD:.17f}")
    print(f"TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    print(f"Business cost:   {total_business_cost:,}")
    print(f"ECE:              {ece:.6f}")
    print("\nArtifacts saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
