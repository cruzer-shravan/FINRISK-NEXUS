from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "modeling" / "paysim"

TARGET = "isFraud"

SPLITS = {
    "train": DATA_DIR / "train.parquet",
    "validation": DATA_DIR / "validation.parquet",
    "test": DATA_DIR / "test.parquet",
}


def main():
    frames = {}

    for name, path in SPLITS.items():
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_parquet(path)
        frames[name] = df

        print(f"\n=== {name.upper()} VALIDATION ===")
        print(f"Rows: {len(df):,}")
        print(f"Step range: {df['step'].min()}–{df['step'].max()}")
        print(f"Fraud count: {int(df[TARGET].sum()):,}")
        print(f"Fraud rate: {df[TARGET].mean():.6%}")
        print(f"Columns: {df.columns.tolist()}")

    # Confirm time periods do not overlap.
    assert frames["train"]["step"].max() < frames["validation"]["step"].min()
    assert frames["validation"]["step"].max() < frames["test"]["step"].min()

    # Confirm all splits have identical schemas.
    schemas = [set(df.columns) for df in frames.values()]
    assert schemas[0] == schemas[1] == schemas[2]

    # Inspect fraud rates in equal-width step intervals.
    all_df = pd.concat(frames.values(), ignore_index=True)

    all_df["step_period"] = pd.cut(
        all_df["step"],
        bins=[0, 100, 200, 300, 400, 500, 600, 700, 743],
        include_lowest=True,
    )

    print("\n=== FRAUD RATE BY TIME PERIOD ===")
    summary = all_df.groupby("step_period", observed=False).agg(
        transactions=(TARGET, "size"),
        fraud_cases=(TARGET, "sum"),
        fraud_rate=(TARGET, "mean"),
    )

    print(summary.to_string())

    # Check whether the overall row counts are preserved.
    total_rows = sum(len(df) for df in frames.values())
    print(f"\nTotal rows across splits: {total_rows:,}")
    assert total_rows == 6_362_620

    print("\nAll structural split checks passed.")
    print("Review temporal fraud rates before model training.")


if __name__ == "__main__":
    main()
