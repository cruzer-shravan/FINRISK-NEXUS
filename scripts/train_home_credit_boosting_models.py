import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
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
from xgboost import XGBClassifier

RANDOM_STATE = 42

TRAIN_PATH = Path("data/modeling/home_credit/application_only/train.parquet")
VALIDATION_PATH = Path("data/modeling/home_credit/application_only/validation.parquet")

MODEL_DIR = Path("models/home_credit/application_only")
REPORT_DIR = Path("reports/home_credit/application_only")

OUTPUTS = [
    MODEL_DIR / "xgboost.joblib",
    MODEL_DIR / "lightgbm.joblib",
    REPORT_DIR / "xgboost_metrics.json",
    REPORT_DIR / "lightgbm_metrics.json",
    REPORT_DIR / "boosting_model_comparison.csv",
]

for path in OUTPUTS:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}. Inspect it before overwriting.")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Home Credit modelling partitions...")

train_df = pd.read_parquet(TRAIN_PATH)
valid_df = pd.read_parquet(VALIDATION_PATH)

TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"

X_train = train_df.drop(columns=[TARGET, ID_COLUMN]).copy()
y_train = train_df[TARGET].astype("int8")

X_valid = valid_df.drop(columns=[TARGET, ID_COLUMN]).copy()
y_valid = valid_df[TARGET].astype("int8")

# Convert Pandas nullable values to NumPy-compatible missing values.
# This prevents pd.NA boolean ambiguity inside sklearn imputers.
numeric_columns = X_train.select_dtypes(include=["number"]).columns
categorical_columns = X_train.select_dtypes(exclude=["number"]).columns

for column in numeric_columns:
    X_train[column] = pd.to_numeric(X_train[column], errors="coerce").astype("float64")

    X_valid[column] = pd.to_numeric(X_valid[column], errors="coerce").astype("float64")

for column in categorical_columns:
    X_train[column] = X_train[column].astype("object")
    X_train[column] = X_train[column].where(pd.notna(X_train[column]), np.nan)

    X_valid[column] = X_valid[column].astype("object")
    X_valid[column] = X_valid[column].where(pd.notna(X_valid[column]), np.nan)

numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()

categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

print(f"Training rows: {len(X_train):,}")
print(f"Validation rows: {len(X_valid):,}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

positive_count = int(y_train.sum())
negative_count = int(len(y_train) - positive_count)
scale_pos_weight = negative_count / positive_count

print(f"Training positive cases: {positive_count:,}")
print(f"Training negative cases: {negative_count:,}")
print(f"Scale positive weight: {scale_pos_weight:.4f}")


def create_preprocessor():
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    missing_values=np.nan,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


print("\nPreprocessing training and validation data...")

preprocessor = create_preprocessor()

X_train_processed = preprocessor.fit_transform(X_train)
X_valid_processed = preprocessor.transform(X_valid)

X_train_processed = csr_matrix(X_train_processed, dtype=np.float32)
X_valid_processed = csr_matrix(X_valid_processed, dtype=np.float32)

print(f"Processed training shape: {X_train_processed.shape}")
print(f"Processed validation shape: {X_valid_processed.shape}")

models = {
    "xgboost": XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=5.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=2,
        random_state=RANDOM_STATE,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=400,
        max_depth=5,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=100,
        reg_lambda=5.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        verbosity=-1,
        n_jobs=2,
        random_state=RANDOM_STATE,
    ),
}


def evaluate_model(model_name, model, X_valid_matrix, y_true):
    probabilities = model.predict_proba(X_valid_matrix)[:, 1]

    precision_values, recall_values, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

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
        "selection_method": "Maximum validation F1",
        "threshold": best_threshold,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_true)),
        "num_numeric_features": len(numeric_features),
        "num_categorical_features": len(categorical_features),
        "processed_feature_count": int(X_train_processed.shape[1]),
        "scale_pos_weight": float(scale_pos_weight),
    }

    return metrics


results = []

for model_name, model in models.items():
    print(f"\n{'=' * 60}")
    print(f"Training {model_name.upper()}...")
    print(f"{'=' * 60}")

    model.fit(X_train_processed, y_train)

    metrics = evaluate_model(
        model_name,
        model,
        X_valid_processed,
        y_valid,
    )

    artifact = {
        "preprocessor": preprocessor,
        "model": model,
    }

    model_path = MODEL_DIR / f"{model_name}.joblib"
    metrics_path = REPORT_DIR / f"{model_name}_metrics.json"

    joblib.dump(artifact, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    results.append(metrics)

    print(f"ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"Average Precision:{metrics['average_precision']:.4f}")
    print(f"Threshold:        {metrics['threshold']:.6f}")
    print(f"Precision:        {metrics['precision']:.4f}")
    print(f"Recall:           {metrics['recall']:.4f}")
    print(f"F1 Score:         {metrics['f1']:.4f}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")
    print(f"Saved model:      {model_path}")
    print(f"Saved metrics:    {metrics_path}")

comparison = pd.DataFrame(results)
comparison.to_csv(
    REPORT_DIR / "boosting_model_comparison.csv",
    index=False,
)

print("\nBoosting model comparison:")
print(
    comparison[
        ["model", "roc_auc", "average_precision", "precision", "recall", "f1", "threshold"]
    ].to_string(index=False)
)

print("\nHome Credit boosting model training completed.")
