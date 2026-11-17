from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/modeling/home_credit/application_only/validation.parquet"
MODEL_DIR = ROOT / "models/home_credit/application_only"
REPORT_DIR = ROOT / "reports/home_credit/application_only"

CANDIDATES = {
    "Tuned Logistic Regression": (MODEL_DIR / "logistic_regression_tuned.joblib"),
    "Tuned XGBoost Sigmoid": (MODEL_DIR / "calibrated_xgboost_tuned_sigmoid.joblib"),
    "Tuned LightGBM Sigmoid": (MODEL_DIR / "calibrated_lightgbm_tuned_sigmoid.joblib"),
}

COST_SCENARIOS = {
    "Equal": {"fp_cost": 1, "fn_cost": 1},
    "Moderate_Default_Cost": {"fp_cost": 1, "fn_cost": 2},
    "High_Default_Cost": {"fp_cost": 1, "fn_cost": 5},
    "Very_High_Default_Cost": {"fp_cost": 1, "fn_cost": 10},
}

DETAILED_PATH = REPORT_DIR / "calibrated_business_cost_threshold_analysis.csv"
SUMMARY_PATH = REPORT_DIR / "calibrated_business_cost_model_summary.csv"

for path in [DETAILED_PATH, SUMMARY_PATH]:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}. Inspect before overwriting.")

for name, path in CANDIDATES.items():
    if not path.exists():
        raise FileNotFoundError(f"{name} artifact not found: {path}")


def prepare_features(frame):
    X = frame.drop(columns=["SK_ID_CURR", "TARGET"]).copy()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")

    for col in categorical_cols:
        X[col] = X[col].astype("object")
        X[col] = X[col].where(pd.notna(X[col]), np.nan)

    return X


def evaluate_all_thresholds(y_true, probabilities, model_name):
    # Evaluate every distinct predicted probability.
    thresholds = np.unique(probabilities)

    # Include boundary thresholds.
    thresholds = np.unique(np.concatenate(([0.0], thresholds, [1.0])))

    order = np.argsort(probabilities)
    sorted_probabilities = probabilities[order]
    sorted_targets = y_true[order]

    # Number of predictions classified positive at each threshold.
    first_positive = np.searchsorted(sorted_probabilities, thresholds, side="left")

    predicted_positive = len(y_true) - first_positive

    # Cumulative positives in sorted order.
    cumulative_positive = np.concatenate(([0], np.cumsum(sorted_targets)))

    true_positive = int(y_true.sum()) - cumulative_positive[first_positive]

    false_positive = predicted_positive - true_positive
    false_negative = int(y_true.sum()) - true_positive
    true_negative = len(y_true) - int(y_true.sum()) - false_positive

    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros(len(thresholds), dtype=float),
        where=predicted_positive != 0,
    )

    recall = true_positive / int(y_true.sum())

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(thresholds), dtype=float),
        where=(precision + recall) != 0,
    )

    f2 = np.divide(
        5 * precision * recall,
        4 * precision + recall,
        out=np.zeros(len(thresholds), dtype=float),
        where=(4 * precision + recall) != 0,
    )

    results = pd.DataFrame(
        {
            "model": model_name,
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
            "tp": true_positive,
        }
    )

    for scenario, costs in COST_SCENARIOS.items():
        results[f"cost_{scenario}"] = (
            false_positive * costs["fp_cost"] + false_negative * costs["fn_cost"]
        )

    return results


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Home Credit validation data...")

    validation = pd.read_parquet(DATA_PATH)
    y_val = validation["TARGET"].astype("int8").to_numpy()
    X_val = prepare_features(validation)

    print(f"Validation rows: {len(validation):,}")
    print(f"Validation features: {X_val.shape[1]}")
    print(f"Default rate: {y_val.mean():.4%}")

    all_details = []
    summaries = []

    for model_name, model_path in CANDIDATES.items():
        print("\n" + "=" * 70)
        print(f"EVALUATING: {model_name}")
        print("=" * 70)

        model = joblib.load(model_path)
        probabilities = model.predict_proba(X_val)[:, 1]

        if not np.isfinite(probabilities).all():
            raise ValueError(f"Non-finite probabilities detected for {model_name}")

        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError(f"Probabilities outside [0, 1] for {model_name}")

        roc_auc = roc_auc_score(y_val, probabilities)
        average_precision = average_precision_score(y_val, probabilities)
        brier = brier_score_loss(y_val, probabilities)

        details = evaluate_all_thresholds(y_val, probabilities, model_name)

        best_f1 = details.loc[details["f1"].idxmax()]
        best_f2 = details.loc[details["f2"].idxmax()]

        summary = {
            "model": model_name,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "brier_score": brier,
            "best_f1_threshold": best_f1["threshold"],
            "best_f1": best_f1["f1"],
            "best_f1_precision": best_f1["precision"],
            "best_f1_recall": best_f1["recall"],
            "best_f2_threshold": best_f2["threshold"],
            "best_f2": best_f2["f2"],
            "best_f2_precision": best_f2["precision"],
            "best_f2_recall": best_f2["recall"],
        }

        for scenario in COST_SCENARIOS:
            cost_column = f"cost_{scenario}"
            best_cost = details.loc[details[cost_column].idxmin()]

            summary[f"best_{scenario}_threshold"] = best_cost["threshold"]
            summary[f"best_{scenario}_cost"] = best_cost[cost_column]

        all_details.append(details)
        summaries.append(summary)

        print(f"ROC-AUC:          {roc_auc:.6f}")
        print(f"Average Precision:{average_precision:.6f}")
        print(f"Brier Score:      {brier:.6f}")
        print(f"Best F1:          {best_f1['f1']:.6f} at threshold {best_f1['threshold']:.6f}")
        print(f"Best F2:          {best_f2['f2']:.6f} at threshold {best_f2['threshold']:.6f}")
        print(f"Thresholds tested: {len(details):,}")

    detailed_df = pd.concat(all_details, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    detailed_df.to_csv(DETAILED_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n" + "=" * 70)
    print("CALIBRATED MODEL BUSINESS-COST SUMMARY")
    print("=" * 70)

    print(summary_df.round(5).to_string(index=False))

    print("\nDetailed results:", DETAILED_PATH)
    print("Model summary:", SUMMARY_PATH)

    print("\nAnalysis completed.")
    print("Only the validation partition was evaluated.")
    print("The test partition was not loaded.")


if __name__ == "__main__":
    main()
