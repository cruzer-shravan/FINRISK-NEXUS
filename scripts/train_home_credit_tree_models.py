import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE = Path("data/modeling/home_credit/application_only")
MODEL_DIR = Path("models/home_credit/application_only")
REPORT_DIR = Path("reports/home_credit/application_only")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "SK_ID_CURR"
TARGET = "TARGET"

RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load training and validation data only
# --------------------------------------------------

print("Loading training and validation datasets...")

train = pd.read_parquet(BASE / "train.parquet")
valid = pd.read_parquet(BASE / "validation.parquet")

X_train = train.drop(columns=[ID_COL, TARGET]).copy()
y_train = train[TARGET].astype("int8")

X_valid = valid.drop(columns=[ID_COL, TARGET]).copy()
y_valid = valid[TARGET].astype("int8")


# --------------------------------------------------
# 3. Identify numerical and categorical features
# --------------------------------------------------

numeric_features = [col for col in X_train.columns if pd.api.types.is_numeric_dtype(X_train[col])]

categorical_features = [col for col in X_train.columns if col not in numeric_features]

assert len(numeric_features) == 104
assert len(categorical_features) == 16

print("\nFeature summary")
print("Numerical:", len(numeric_features))
print("Categorical:", len(categorical_features))


# --------------------------------------------------
# 4. Convert nullable Pandas values
# --------------------------------------------------

for col in numeric_features:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce").astype("float32")

    X_valid[col] = pd.to_numeric(X_valid[col], errors="coerce").astype("float32")

for col in categorical_features:
    X_train[col] = X_train[col].astype(object).where(X_train[col].notna(), np.nan)

    X_valid[col] = X_valid[col].astype(object).where(X_valid[col].notna(), np.nan)


# --------------------------------------------------
# 5. Preprocessing pipeline
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", dtype=np.float32)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
    sparse_threshold=1.0,
)


# --------------------------------------------------
# 6. Define candidate models
# --------------------------------------------------

candidates = {
    "decision_tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=100,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=2,
    ),
}


# --------------------------------------------------
# 7. Train and evaluate each model
# --------------------------------------------------

comparison = []

for model_name, classifier in candidates.items():
    print("\n" + "=" * 65)
    print("TRAINING:", model_name.upper())
    print("=" * 65)

    model_path = MODEL_DIR / f"{model_name}.joblib"
    metrics_path = REPORT_DIR / f"{model_name}_metrics.json"

    if model_path.exists() or metrics_path.exists():
        raise FileExistsError(
            f"Existing artifact found for {model_name}. "
            "Inspect it before deciding whether to overwrite."
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    print("Fitting model...")
    pipeline.fit(X_train, y_train)

    print("Generating validation probabilities...")
    probabilities = pipeline.predict_proba(X_valid)[:, 1]

    roc_auc = roc_auc_score(y_valid, probabilities)
    average_precision = average_precision_score(y_valid, probabilities)

    # Select threshold using validation F1 only.
    precision_values, recall_values, thresholds = precision_recall_curve(y_valid, probabilities)

    f1_values = (
        2
        * precision_values[:-1]
        * recall_values[:-1]
        / (precision_values[:-1] + recall_values[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

    predictions = (probabilities >= best_threshold).astype("int8")

    metrics = {
        "model": model_name,
        "training_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "numeric_features": int(len(numeric_features)),
        "categorical_features": int(len(categorical_features)),
        "threshold_selection": "Maximum validation F1",
        "threshold": best_threshold,
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_valid, predictions).tolist(),
        "random_state": RANDOM_STATE,
    }

    joblib.dump(pipeline, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    comparison.append(
        {
            "model": model_name,
            "threshold": metrics["threshold"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
    )

    print("\nVALIDATION RESULTS")
    print("ROC-AUC:", round(roc_auc, 4))
    print("Average Precision:", round(average_precision, 4))
    print("Threshold:", round(best_threshold, 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1:", round(metrics["f1"], 4))
    print("Confusion Matrix:", metrics["confusion_matrix"])

    print("Model saved:", model_path)
    print("Metrics saved:", metrics_path)


# --------------------------------------------------
# 8. Save model comparison
# --------------------------------------------------

comparison_df = pd.DataFrame(comparison)

comparison_path = REPORT_DIR / "tree_model_comparison.csv"

comparison_df.to_csv(comparison_path, index=False)

print("\n" + "=" * 65)
print("TREE MODEL COMPARISON")
print("=" * 65)

print(comparison_df.to_string(index=False))

print("\nComparison saved:", comparison_path)
print("\nTREE MODEL TRAINING COMPLETED.")
print("Test dataset was not accessed.")
