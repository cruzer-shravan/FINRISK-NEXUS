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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE = Path("data/modeling/home_credit/application_only")
MODEL_DIR = Path("models/home_credit/application_only")
REPORT_DIR = Path("reports/home_credit/application_only")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
METRICS_PATH = REPORT_DIR / "logistic_regression_metrics.json"

ID_COL = "SK_ID_CURR"
TARGET = "TARGET"


# --------------------------------------------------
# 2. Load training and validation data only
# --------------------------------------------------

print("Loading training and validation datasets...")

train = pd.read_parquet(BASE / "train.parquet")
valid = pd.read_parquet(BASE / "validation.parquet")

assert list(train.columns) == list(valid.columns)
assert train[TARGET].notna().all()
assert valid[TARGET].notna().all()

X_train = train.drop(columns=[ID_COL, TARGET]).copy()
y_train = train[TARGET].astype("int8")

X_valid = valid.drop(columns=[ID_COL, TARGET]).copy()
y_valid = valid[TARGET].astype("int8")

# Convert Pandas nullable dtypes into sklearn-compatible NumPy values.
# Keep missing values as np.nan; do not impute outside the pipeline.

numeric_source_features = [
    col for col in X_train.columns if pd.api.types.is_numeric_dtype(X_train[col])
]

categorical_source_features = [col for col in X_train.columns if col not in numeric_source_features]

for col in numeric_source_features:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce").astype("float64")

    X_valid[col] = pd.to_numeric(X_valid[col], errors="coerce").astype("float64")

for col in categorical_source_features:
    X_train[col] = X_train[col].astype(object).where(X_train[col].notna(), np.nan)

    X_valid[col] = X_valid[col].astype(object).where(X_valid[col].notna(), np.nan)


# --------------------------------------------------
# 3. Identify feature types
# --------------------------------------------------

numeric_features = [col for col in X_train.columns if pd.api.types.is_numeric_dtype(X_train[col])]

categorical_features = [col for col in X_train.columns if col not in numeric_features]

print("\nFeature summary")
print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

assert len(numeric_features) == 104
assert len(categorical_features) == 16


# --------------------------------------------------
# 4. Build preprocessing pipelines
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)


# --------------------------------------------------
# 5. Build Logistic Regression model
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=500,
                tol=0.001,
                random_state=42,
            ),
        ),
    ]
)


# --------------------------------------------------
# 6. Train model
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 7. Validation predictions
# --------------------------------------------------

print("\nEvaluating on validation data...")

y_valid_proba = model.predict_proba(X_valid)[:, 1]

roc_auc = roc_auc_score(y_valid, y_valid_proba)
average_precision = average_precision_score(y_valid, y_valid_proba)


# --------------------------------------------------
# 8. Select threshold using validation F1
# --------------------------------------------------

precision_values, recall_values, thresholds = precision_recall_curve(y_valid, y_valid_proba)

f1_values = (
    2
    * precision_values[:-1]
    * recall_values[:-1]
    / (precision_values[:-1] + recall_values[:-1] + 1e-12)
)

best_index = int(f1_values.argmax())
best_threshold = float(thresholds[best_index])

y_valid_pred = (y_valid_proba >= best_threshold).astype("int8")


# --------------------------------------------------
# 9. Calculate validation metrics
# --------------------------------------------------

metrics = {
    "model": "LogisticRegression",
    "dataset": "Home Credit Application Only",
    "training_rows": int(len(train)),
    "validation_rows": int(len(valid)),
    "numeric_features": int(len(numeric_features)),
    "categorical_features": int(len(categorical_features)),
    "class_weight": "balanced",
    "solver": "lbfgs",
    "threshold_selection": "Maximum validation F1",
    "threshold": best_threshold,
    "roc_auc": float(roc_auc),
    "average_precision": float(average_precision),
    "precision": float(precision_score(y_valid, y_valid_pred, zero_division=0)),
    "recall": float(recall_score(y_valid, y_valid_pred, zero_division=0)),
    "f1": float(f1_score(y_valid, y_valid_pred, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_valid, y_valid_pred).tolist(),
    "classification_report": classification_report(
        y_valid,
        y_valid_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    ),
}


# --------------------------------------------------
# 10. Save model and metrics
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nVALIDATION RESULTS")
print("------------------------------")
print("ROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(average_precision, 4))
print("Selected Threshold:", round(best_threshold, 4))
print("Precision:", round(metrics["precision"], 4))
print("Recall:", round(metrics["recall"], 4))
print("F1 Score:", round(metrics["f1"], 4))
print("Confusion Matrix:", metrics["confusion_matrix"])

print("\nModel saved:", MODEL_PATH)
print("Metrics saved:", METRICS_PATH)

print("\nLOGISTIC REGRESSION BASELINE COMPLETED.")
print("Test dataset was not accessed.")
