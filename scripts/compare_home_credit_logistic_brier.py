import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DATA_PATH = "data/modeling/home_credit/application_only/validation.parquet"
MODEL_DIR = "models/home_credit/application_only"

validation = pd.read_parquet(DATA_PATH)

X_val = validation.drop(columns=["SK_ID_CURR", "TARGET"]).copy()
y_val = validation["TARGET"]

# Convert Pandas nullable values into NumPy-compatible missing values.
for col in X_val.columns:
    if is_numeric_dtype(X_val[col].dtype):
        X_val[col] = X_val[col].astype("float64")
    else:
        X_val[col] = X_val[col].astype("object")
        X_val[col] = X_val[col].where(X_val[col].notna(), np.nan)

models = {
    "Original Logistic Regression": f"{MODEL_DIR}/logistic_regression.joblib",
    "Tuned Logistic Regression": f"{MODEL_DIR}/logistic_regression_tuned.joblib",
}

print("\nLOGISTIC REGRESSION — VALIDATION COMPARISON")
print("=" * 65)

for name, path in models.items():
    model = joblib.load(path)
    probabilities = model.predict_proba(X_val)[:, 1]

    print(f"\n{name}")
    print("-" * 45)
    print(f"Brier Score:       {brier_score_loss(y_val, probabilities):.6f}")
    print(f"ROC-AUC:           {roc_auc_score(y_val, probabilities):.6f}")
    print(f"Average Precision: {average_precision_score(y_val, probabilities):.6f}")

prevalence = y_val.mean()
reference_brier = brier_score_loss(y_val, np.full(len(y_val), prevalence))

print("\nConstant-prevalence reference:")
print(f"Default prevalence: {prevalence:.6f}")
print(f"Reference Brier:    {reference_brier:.6f}")
