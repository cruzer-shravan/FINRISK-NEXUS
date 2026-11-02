import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = PROJECT_ROOT / "data" / "gold" / "paysim" / "transaction_features.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "modeling" / "paysim"

TARGET = "isFraud"

TEMPORAL_COLUMN = "step"

MODEL_NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
]

MODEL_CATEGORICAL_FEATURES = [
    "type",
]

MODEL_FEATURES = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES

DATASET_COLUMNS = [
    TEMPORAL_COLUMN,
    *MODEL_FEATURES,
]

# Chronological targets based on transaction counts.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15


def main():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Gold dataset not found: {SOURCE_PATH}")

    df = pd.read_parquet(SOURCE_PATH)

    required = DATASET_COLUMNS + [TARGET]
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Required columns missing from Gold data: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values found in required modeling columns.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    if not df[TEMPORAL_COLUMN].is_monotonic_increasing:
        # Sorting is safe because the time index defines the split.
        df = df.sort_values(TEMPORAL_COLUMN, kind="stable").reset_index(drop=True)

    # Calculate split boundaries using whole time steps.
    # Transactions from the same step cannot cross splits.
    step_counts = df.groupby(TEMPORAL_COLUMN, sort=True).size()
    cumulative_counts = step_counts.cumsum().to_numpy()
    steps = step_counts.index.to_numpy()

    total_rows = len(df)

    train_cutoff = int(total_rows * TRAIN_FRACTION)
    validation_cutoff = int(total_rows * (TRAIN_FRACTION + VALIDATION_FRACTION))

    train_idx = min(
        np.searchsorted(cumulative_counts, train_cutoff),
        len(steps) - 1,
    )
    validation_idx = min(
        np.searchsorted(cumulative_counts, validation_cutoff),
        len(steps) - 1,
    )

    train_end_step = int(steps[train_idx])
    validation_end_step = int(steps[validation_idx])

    if train_end_step >= validation_end_step:
        raise ValueError("Invalid chronological split boundaries.")

    train = df.loc[df[TEMPORAL_COLUMN] <= train_end_step, required].copy()

    validation = df.loc[
        (df[TEMPORAL_COLUMN] > train_end_step) & (df[TEMPORAL_COLUMN] <= validation_end_step),
        required,
    ].copy()

    test = df.loc[df[TEMPORAL_COLUMN] > validation_end_step, required].copy()

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    if any(part.empty for part in splits.values()):
        raise ValueError("At least one split is empty.")

    # Require both target classes in each split.
    for name, part in splits.items():
        if part[TARGET].nunique() != 2:
            raise ValueError(
                f"{name} split does not contain both target classes. Review the time boundaries."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, part in splits.items():
        output_path = OUTPUT_DIR / f"{name}.parquet"
        part.to_parquet(output_path, index=False)

    manifest = {
        "dataset": "PaySim fraud modeling dataset",
        "source": str(SOURCE_PATH.relative_to(PROJECT_ROOT)),
        "target": TARGET,
        "temporal_column": TEMPORAL_COLUMN,
        "model_features": MODEL_FEATURES,
        "numeric_features": MODEL_NUMERIC_FEATURES,
        "categorical_features": MODEL_CATEGORICAL_FEATURES,
        "dataset_columns": DATASET_COLUMNS,
        "feature_usage": {
            "temporal_column": (
                "Used for chronological splitting and temporal analysis; not passed to the model."
            ),
            "model_features": ("Predictor columns passed to the machine-learning estimator."),
        },
        "excluded_columns": {
            "nameOrig": "Identifier excluded from baseline",
            "nameDest": "Identifier excluded from baseline",
            "newbalanceOrig": "Post-transaction information",
            "newbalanceDest": "Post-transaction information",
            "isFlaggedFraud": "Existing rule-based flag",
            "origin_balance_error": "Requires timing/formula review",
            "destination_balance_error": "Requires timing/formula review",
            "origin_balance_mismatch": "Requires timing/formula review",
            "destination_balance_mismatch": "Requires timing/formula review",
        },
        "split_method": "Chronological by whole step",
        "target_row_fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": (1.0 - TRAIN_FRACTION - VALIDATION_FRACTION),
        },
        "boundaries": {
            "train_max_step": train_end_step,
            "validation_min_step": train_end_step + 1,
            "validation_max_step": validation_end_step,
            "test_min_step": validation_end_step + 1,
        },
        "splits": {},
    }

    print("\n=== PAYSIM MODELING DATASET ===")
    print(f"Source rows: {total_rows:,}")
    print(f"Dataset columns: {DATASET_COLUMNS}")
    print(f"Model features: {MODEL_FEATURES}")
    print(f"Temporal column: {TEMPORAL_COLUMN}")
    print(f"Target: {TARGET}")

    for name, part in splits.items():
        fraud_count = int(part[TARGET].sum())

        manifest["splits"][name] = {
            "rows": len(part),
            "fraud_count": fraud_count,
            "fraud_rate": float(part[TARGET].mean()),
            "min_step": int(part[TEMPORAL_COLUMN].min()),
            "max_step": int(part[TEMPORAL_COLUMN].max()),
            "columns": list(part.columns),
        }

        print(f"\n{name.upper()}")
        print(f"Rows: {len(part):,}")
        print(
            f"{TEMPORAL_COLUMN} range: {part[TEMPORAL_COLUMN].min()}–{part[TEMPORAL_COLUMN].max()}"
        )
        print(f"Fraud cases: {fraud_count:,}")
        print(f"Fraud rate: {part[TARGET].mean():.6%}")

    manifest_path = OUTPUT_DIR / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")
    print("\nModeling dataset preparation completed.")


if __name__ == "__main__":
    main()
