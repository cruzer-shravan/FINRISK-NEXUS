from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DATA_PATH = "data/modeling/home_credit/application_only/validation.parquet"
MODEL_DIR = Path("models/home_credit/application_only")
REPORT_DIR = Path("reports/home_credit/application_only")

validation = pd.read_parquet(DATA_PATH)

X_val = validation.drop(columns=["SK_ID_CURR", "TARGET"]).copy()
y_val = validation["TARGET"]

for col in X_val.columns:
    if is_numeric_dtype(X_val[col].dtype):
        X_val[col] = X_val[col].astype("float64")
    else:
        X_val[col] = X_val[col].astype("object")
        X_val[col] = X_val[col].where(X_val[col].notna(), np.nan)

models = {
    "Tuned Logistic Regression": "logistic_regression_tuned.joblib",
    "Tuned XGBoost": "xgboost_tuned.joblib",
    "Baseline LightGBM": "lightgbm.joblib",
    "Tuned LightGBM": "lightgbm_tuned.joblib",
}

summary = []
plt.figure(figsize=(10, 8))

for name, filename in models.items():
    model = joblib.load(MODEL_DIR / filename)
    probabilities = model.predict_proba(X_val)[:, 1]

    brier = brier_score_loss(y_val, probabilities)
    roc_auc = roc_auc_score(y_val, probabilities)
    ap = average_precision_score(y_val, probabilities)

    summary.append(
        {
            "model": name,
            "brier_score": brier,
            "roc_auc": roc_auc,
            "average_precision": ap,
        }
    )

    fraction_positive, mean_predicted = calibration_curve(
        y_val, probabilities, n_bins=10, strategy="quantile"
    )

    plt.plot(mean_predicted, fraction_positive, marker="o", linewidth=2, label=name)

plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
plt.xlabel("Mean predicted default probability")
plt.ylabel("Observed default rate")
plt.title("Home Credit — Model Reliability Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

chart_path = REPORT_DIR / "model_reliability_comparison.png"
plt.savefig(chart_path, dpi=160)
plt.close()

summary_df = pd.DataFrame(summary).sort_values("brier_score")

csv_path = REPORT_DIR / "model_reliability_comparison.csv"
summary_df.to_csv(csv_path, index=False)

print("\nMODEL RELIABILITY COMPARISON")
print("=" * 75)
print(summary_df.to_string(index=False))

print("\nSaved chart:", chart_path)
print("Saved metrics:", csv_path)
