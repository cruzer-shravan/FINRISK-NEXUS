from pathlib import Path

import joblib

model_dir = Path("models/home_credit/application_only")

files = [
    "logistic_regression_tuned.joblib",
    "xgboost_tuned.joblib",
    "lightgbm.joblib",
    "lightgbm_tuned.joblib",
]

for filename in files:
    path = model_dir / filename
    obj = joblib.load(path)

    print("\nMODEL:", filename)
    print("TYPE:", type(obj))

    if isinstance(obj, dict):
        print("KEYS:", list(obj.keys()))
        for key, value in obj.items():
            print(f"  {key}: {type(value)}")
    else:
        print("PIPELINE STEPS:", list(obj.named_steps.keys()))
