import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_ITER = 10

TRAIN_PATH = Path("data/modeling/home_credit/application_only/train.parquet")
VALIDATION_PATH = Path("data/modeling/home_credit/application_only/validation.parquet")

MODEL_DIR = Path("models/home_credit/application_only")
REPORT_DIR = Path("reports/home_credit/application_only")

OUTPUTS = [
    MODEL_DIR / "xgboost_tuned.joblib",
    MODEL_DIR / "lightgbm_tuned.joblib",
    REPORT_DIR / "xgboost_tuned_metrics.json",
    REPORT_DIR / "lightgbm_tuned_metrics.json",
    REPORT_DIR / "xgboost_tuning_results.csv",
    REPORT_DIR / "lightgbm_tuning_results.csv",
    REPORT_DIR / "tuned_model_comparison.csv",
]

for path in OUTPUTS:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}. Inspect before overwriting.")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading training and validation partitions...")

train_df = pd.read_parquet(TRAIN_PATH)
valid_df = pd.read_parquet(VALIDATION_PATH)

TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"

X_train = train_df.drop(columns=[TARGET, ID_COLUMN]).copy()
y_train = train_df[TARGET].astype("int8")

X_valid = valid_df.drop(columns=[TARGET, ID_COLUMN]).copy()
y_valid = valid_df[TARGET].astype("int8")

# Convert Pandas nullable values to NumPy-compatible missing values.
numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()

categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

for column in numeric_features:
    X_train[column] = pd.to_numeric(X_train[column], errors="coerce").astype("float64")

    X_valid[column] = pd.to_numeric(X_valid[column], errors="coerce").astype("float64")

for column in categorical_features:
    X_train[column] = X_train[column].astype("object")
    X_train[column] = X_train[column].where(pd.notna(X_train[column]), np.nan)

    X_valid[column] = X_valid[column].astype("object")
    X_valid[column] = X_valid[column].where(pd.notna(X_valid[column]), np.nan)

positive_count = int(y_train.sum())
negative_count = int(len(y_train) - positive_count)
imbalance_ratio = negative_count / positive_count

print(f"Training rows: {len(X_train):,}")
print(f"Validation rows: {len(X_valid):,}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Training imbalance ratio: {imbalance_ratio:.4f}")


def create_pipeline(model):
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


models_and_parameters = {
    "xgboost": (
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
        {
            "model__n_estimators": [300, 500],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.03, 0.05, 0.07],
            "model__min_child_weight": [5, 10, 20],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [2.0, 5.0, 10.0],
            "model__scale_pos_weight": [6.0, 9.0, imbalance_ratio, 14.0],
        },
    ),
    "lightgbm": (
        LGBMClassifier(
            objective="binary",
            verbosity=-1,
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
        {
            "model__n_estimators": [300, 500],
            "model__num_leaves": [15, 31, 47],
            "model__max_depth": [4, 5, 6],
            "model__learning_rate": [0.03, 0.05, 0.07],
            "model__min_child_samples": [50, 100, 150],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [2.0, 5.0, 10.0],
            "model__scale_pos_weight": [6.0, 9.0, imbalance_ratio, 14.0],
            "model__subsample_freq": [1],
        },
    ),
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE,
)


def evaluate_on_validation(model_name, fitted_pipeline, best_cv_score):
    probabilities = fitted_pipeline.predict_proba(X_valid)[:, 1]

    precision_values, recall_values, thresholds = precision_recall_curve(
        y_valid,
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

    return {
        "model": model_name,
        "selection_metric": "Cross-validation Average Precision",
        "best_cv_average_precision": float(best_cv_score),
        "validation_threshold": best_threshold,
        "validation_roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "validation_average_precision": float(average_precision_score(y_valid, probabilities)),
        "validation_precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "validation_recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "validation_f1": float(f1_score(y_valid, predictions, zero_division=0)),
        "validation_confusion_matrix": confusion_matrix(y_valid, predictions).tolist(),
    }


all_results = []

for model_name, (estimator, parameter_space) in models_and_parameters.items():
    print("\n" + "=" * 65)
    print(f"HYPERPARAMETER TUNING: {model_name.upper()}")
    print("=" * 65)

    pipeline = create_pipeline(estimator)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space,
        n_iter=N_ITER,
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=1,
        verbose=2,
        random_state=RANDOM_STATE,
        error_score="raise",
        return_train_score=False,
    )

    search.fit(X_train, y_train)

    print(f"\nBest CV Average Precision: {search.best_score_:.6f}")
    print("Best hyperparameters:")

    for parameter, value in search.best_params_.items():
        print(f"  {parameter}: {value}")

    metrics = evaluate_on_validation(
        model_name,
        search.best_estimator_,
        search.best_score_,
    )

    model_path = MODEL_DIR / f"{model_name}_tuned.joblib"
    metrics_path = REPORT_DIR / f"{model_name}_tuned_metrics.json"
    tuning_path = REPORT_DIR / f"{model_name}_tuning_results.csv"

    joblib.dump(search.best_estimator_, model_path)

    metrics["best_parameters"] = search.best_params_

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(tuning_path, index=False)

    all_results.append(metrics)

    print("\nValidation results:")
    print(f"ROC-AUC:          {metrics['validation_roc_auc']:.4f}")
    print(f"Average Precision:{metrics['validation_average_precision']:.4f}")
    print(f"Threshold:        {metrics['validation_threshold']:.6f}")
    print(f"Precision:        {metrics['validation_precision']:.4f}")
    print(f"Recall:           {metrics['validation_recall']:.4f}")
    print(f"F1 Score:         {metrics['validation_f1']:.4f}")
    print(f"Confusion Matrix: {metrics['validation_confusion_matrix']}")

    print(f"Saved model:      {model_path}")
    print(f"Saved metrics:    {metrics_path}")
    print(f"Saved CV results: {tuning_path}")

comparison = pd.DataFrame(all_results)

comparison.to_csv(
    REPORT_DIR / "tuned_model_comparison.csv",
    index=False,
)

print("\n" + "=" * 65)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 65)

print(
    comparison[
        [
            "model",
            "best_cv_average_precision",
            "validation_roc_auc",
            "validation_average_precision",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "validation_threshold",
        ]
    ].to_string(index=False)
)

print("\nThe test partition was not loaded or evaluated.")
