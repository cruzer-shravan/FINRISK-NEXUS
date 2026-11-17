import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/modeling/home_credit/application_only/validation.parquet"
MODEL_DIR = ROOT / "models/home_credit/application_only"
REPORT_DIR = ROOT / "reports/home_credit/application_only"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "XGBoost Tuned": {
        "model": MODEL_DIR / "xgboost_tuned.joblib",
        "metrics": REPORT_DIR / "xgboost_tuned_metrics.json",
    },
    "LightGBM Baseline": {
        "model": MODEL_DIR / "lightgbm.joblib",
        "metrics": REPORT_DIR / "lightgbm_metrics.json",
    },
    "LightGBM Tuned": {
        "model": MODEL_DIR / "lightgbm_tuned.joblib",
        "metrics": REPORT_DIR / "lightgbm_tuned_metrics.json",
    },
}

BASE_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

COST_SCENARIOS = {
    "Equal": {"fp_cost": 1, "fn_cost": 1},
    "Moderate_Default_Cost": {"fp_cost": 1, "fn_cost": 2},
    "High_Default_Cost": {"fp_cost": 1, "fn_cost": 5},
    "Very_High_Default_Cost": {"fp_cost": 1, "fn_cost": 10},
}


def prepare_features(frame):
    """Convert nullable Pandas values for sklearn compatibility."""
    X = frame.drop(columns=["SK_ID_CURR", "TARGET"]).copy()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].astype("float64")

    if categorical_cols:
        X[categorical_cols] = (
            X[categorical_cols].astype("object").where(X[categorical_cols].notna(), np.nan)
        )

    return X


def get_probabilities(artifact, X):
    """Support both baseline dictionaries and fitted sklearn pipelines."""
    if isinstance(artifact, dict):
        preprocessor = artifact["preprocessor"]
        model = artifact["model"]

        X_transformed = preprocessor.transform(X)
        return model.predict_proba(X_transformed)[:, 1]

    return artifact.predict_proba(X)[:, 1]


def evaluate_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    f2 = fbeta_score(y_true, predictions, beta=2, zero_division=0)

    result = {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    for scenario, costs in COST_SCENARIOS.items():
        result[f"cost_{scenario}"] = fp * costs["fp_cost"] + fn * costs["fn_cost"]

    return result


def main():
    print("Loading Home Credit validation data...")

    validation = pd.read_parquet(DATA_PATH)
    y_val = validation["TARGET"].astype(int).to_numpy()
    X_val = prepare_features(validation)

    print(f"Validation rows: {len(validation):,}")
    print(f"Validation features: {X_val.shape[1]}")
    print(f"Default rate: {y_val.mean():.4%}")

    all_results = []
    model_summary = []

    for model_name, paths in CANDIDATES.items():
        print(f"\nEvaluating: {model_name}")

        artifact = joblib.load(paths["model"])

        with open(paths["metrics"], encoding="utf-8") as f:
            saved_metrics = json.load(f)

        probabilities = get_probabilities(artifact, X_val)

        if not np.isfinite(probabilities).all():
            raise ValueError(f"Non-finite probabilities detected for {model_name}")

        native_threshold = saved_metrics.get("threshold")

        thresholds = set(BASE_THRESHOLDS)
        if native_threshold is not None:
            thresholds.add(float(native_threshold))

        thresholds = sorted(thresholds)

        roc_auc = roc_auc_score(y_val, probabilities)
        average_precision = average_precision_score(y_val, probabilities)
        brier = brier_score_loss(y_val, probabilities)

        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Average Precision: {average_precision:.4f}")
        print(f"Brier Score: {brier:.4f}")
        print(f"Saved validation threshold: {native_threshold}")

        model_rows = []

        for threshold in thresholds:
            result = evaluate_threshold(y_val, probabilities, threshold)
            result["model"] = model_name
            model_rows.append(result)
            all_results.append(result)

        best_f1 = max(model_rows, key=lambda row: row["f1"])

        summary = {
            "model": model_name,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "brier_score": brier,
            "saved_threshold": native_threshold,
            "best_f1_threshold": best_f1["threshold"],
            "best_f1": best_f1["f1"],
        }

        for scenario in COST_SCENARIOS:
            best_cost = min(model_rows, key=lambda row: row[f"cost_{scenario}"])
            summary[f"best_{scenario}_threshold"] = best_cost["threshold"]
            summary[f"best_{scenario}_cost"] = best_cost[f"cost_{scenario}"]

        model_summary.append(summary)

    detailed_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(model_summary)

    detailed_path = REPORT_DIR / "business_cost_threshold_analysis.csv"
    summary_path = REPORT_DIR / "business_cost_model_summary.csv"

    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n========== MODEL SUMMARY ==========")
    print(summary_df.round(4).to_string(index=False))

    print("\n========== COST ANALYSIS ==========")
    for scenario in COST_SCENARIOS:
        cost_col = f"best_{scenario}_cost"
        threshold_col = f"best_{scenario}_threshold"

        print(f"\n{scenario}:")
        print(summary_df[["model", threshold_col, cost_col]].to_string(index=False))

    print("\nAnalysis completed successfully.")
    print(f"Detailed results: {detailed_path}")
    print(f"Model summary: {summary_path}")


if __name__ == "__main__":
    main()
