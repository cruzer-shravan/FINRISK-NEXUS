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
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data/modeling/home_credit/application_only"
MODEL_DIR = ROOT / "models/home_credit/application_only"
REPORT_DIR = ROOT / "reports/home_credit/application_only"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"

MODEL_PATH = MODEL_DIR / "logistic_regression_tuned.joblib"
METRICS_PATH = REPORT_DIR / "logistic_regression_tuned_metrics.json"
SEARCH_RESULTS_PATH = REPORT_DIR / "logistic_regression_tuning_results.csv"


def prepare_features(frame):
    X = frame.drop(columns=["SK_ID_CURR", "TARGET"]).copy()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].astype("float64")

    if categorical_cols:
        X[categorical_cols] = (
            X[categorical_cols].astype("object").where(X[categorical_cols].notna(), np.nan)
        )

    return X, numeric_cols, categorical_cols


def main():
    print("Loading Home Credit training and validation data...")

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)

    y_train = train["TARGET"].astype(int)
    y_val = validation["TARGET"].astype(int).to_numpy()

    X_train, numeric_cols, categorical_cols = prepare_features(train)
    X_val, _, _ = prepare_features(validation)

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Training default rate: {y_train.mean():.4%}")

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    max_iter=1000,
                    tol=0.001,
                    random_state=42,
                ),
            ),
        ]
    )

    parameter_distributions = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.0, 1.0],
        "model__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=42,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=6,
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=1,
        verbose=2,
        random_state=42,
        return_train_score=True,
        error_score="raise",
    )

    print("\nStarting Logistic Regression hyperparameter tuning...")
    print("6 configurations x 2 folds = 12 model fits.")

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print("\nBest cross-validation Average Precision:")
    print(f"{search.best_score_:.6f}")

    print("\nBest hyperparameters:")
    print(json.dumps(search.best_params_, indent=4))

    print("\nEvaluating the selected model on validation data...")

    probabilities = best_model.predict_proba(X_val)[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite validation probabilities detected.")

    # Select threshold using validation F1.
    from sklearn.metrics import precision_recall_curve

    precision_values, recall_values, thresholds = precision_recall_curve(y_val, probabilities)

    f1_values = (
        2
        * precision_values[:-1]
        * recall_values[:-1]
        / (precision_values[:-1] + recall_values[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

    predictions = (probabilities >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val, predictions, labels=[0, 1]).ravel()

    metrics = {
        "model": "LogisticRegression_Tuned",
        "selection_strategy": "RandomizedSearchCV",
        "primary_cv_metric": "average_precision",
        "best_cv_average_precision": float(search.best_score_),
        "best_parameters": search.best_params_,
        "cv_folds": 2,
        "random_search_iterations": 6,
        "validation_roc_auc": float(roc_auc_score(y_val, probabilities)),
        "validation_average_precision": float(average_precision_score(y_val, probabilities)),
        "validation_brier_score": float(brier_score_loss(y_val, probabilities)),
        "threshold": best_threshold,
        "precision": float(precision_score(y_val, predictions, zero_division=0)),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "validation_rows": int(len(validation)),
        "test_data_used": False,
    }

    joblib.dump(best_model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results = cv_results.sort_values("rank_test_score")
    cv_results.to_csv(SEARCH_RESULTS_PATH, index=False)

    print("\n========== TUNED LOGISTIC REGRESSION RESULTS ==========")
    print(f"ROC-AUC: {metrics['validation_roc_auc']:.4f}")
    print(f"Average Precision: {metrics['validation_average_precision']:.4f}")
    print(f"Brier Score: {metrics['validation_brier_score']:.4f}")
    print(f"Threshold: {best_threshold:.6f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(f"TN: {tn:,} | FP: {fp:,}")
    print(f"FN: {fn:,} | TP: {tp:,}")

    print("\n========== SAVED ARTIFACTS ==========")
    print(MODEL_PATH)
    print(METRICS_PATH)
    print(SEARCH_RESULTS_PATH)

    print("\nLogistic Regression tuning completed successfully.")


if __name__ == "__main__":
    main()
