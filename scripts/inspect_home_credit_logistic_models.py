import joblib

paths = [
    "models/home_credit/application_only/logistic_regression.joblib",
    "models/home_credit/application_only/logistic_regression_tuned.joblib",
]

for path in paths:
    obj = joblib.load(path)

    print("\nFILE:", path)
    print("OBJECT TYPE:", type(obj))
    print("DICT KEYS:", list(obj.keys()) if isinstance(obj, dict) else "N/A")

    if hasattr(obj, "named_steps") and "model" in obj.named_steps:
        model = obj.named_steps["model"]
        print("MODEL TYPE:", type(model))
        print("MODEL ITERATIONS:", getattr(model, "n_iter_", "Not available"))

    elif isinstance(obj, dict):
        for key, value in obj.items():
            if hasattr(value, "n_iter_"):
                print("MODEL:", key)
                print("MODEL ITERATIONS:", value.n_iter_)
