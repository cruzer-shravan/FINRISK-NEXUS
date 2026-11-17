from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.calibration import calibration_curve

DATA_PATH = "data/modeling/home_credit/application_only/validation.parquet"
MODEL_DIR = Path("models/home_credit/application_only")
REPORT_DIR = Path("reports/home_credit/application_only")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

validation = pd.read_parquet(DATA_PATH)

X_val = validation.drop(columns=["SK_ID_CURR", "TARGET"]).copy()
y_val = validation["TARGET"]

# Make missing values compatible with the saved pipelines.
for col in X_val.columns:
    if is_numeric_dtype(X_val[col].dtype):
        X_val[col] = X_val[col].astype("float64")
    else:
        X_val[col] = X_val[col].astype("object")
        X_val[col] = X_val[col].where(X_val[col].notna(), np.nan)

models = {
    "Original Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Tuned Logistic Regression": MODEL_DIR / "logistic_regression_tuned.joblib",
}

results = []
plt.figure(figsize=(9, 7))

for name, path in models.items():
    model = joblib.load(path)
    probabilities = model.predict_proba(X_val)[:, 1]

    fraction_positive, mean_predicted = calibration_curve(
        y_val, probabilities, n_bins=10, strategy="quantile"
    )

    plt.plot(mean_predicted, fraction_positive, marker="o", linewidth=2, label=name)

    for i, (predicted, observed) in enumerate(
        zip(mean_predicted, fraction_positive, strict=True), start=1
    ):
        results.append(
            {
                "model": name,
                "bin": i,
                "mean_predicted_probability": predicted,
                "observed_default_rate": observed,
                "calibration_gap": observed - predicted,
            }
        )

plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.xlabel("Mean predicted default probability")
plt.ylabel("Observed default rate")
plt.title("Home Credit — Logistic Regression Reliability Diagram")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

chart_path = REPORT_DIR / "logistic_regression_reliability.png"
plt.savefig(chart_path, dpi=160)
plt.close()

results_df = pd.DataFrame(results)

csv_path = REPORT_DIR / "logistic_regression_reliability.csv"
results_df.to_csv(csv_path, index=False)

print("\nRELIABILITY ANALYSIS COMPLETED")
print("=" * 65)
print(results_df.to_string(index=False))

print("\nSaved chart:", chart_path)
print("Saved data:", csv_path)
