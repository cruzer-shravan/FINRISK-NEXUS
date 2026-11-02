from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "gold" / "paysim" / "transaction_features.parquet"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Gold dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    print("\n=== DATASET OVERVIEW ===")
    print(f"File: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n=== SCHEMA ===")
    print(df.dtypes.to_string())

    print("\n=== COLUMN NAMES ===")
    print(df.columns.tolist())

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum().to_string())

    print("\n=== TARGET DISTRIBUTION ===")
    if "isFraud" in df.columns:
        print(df["isFraud"].value_counts(dropna=False).to_string())
        print("\nTarget proportions:")
        print(df["isFraud"].value_counts(normalize=True).to_string())
    else:
        print("isFraud not found; inspect the actual target column.")

    print("\n=== TIME RANGE ===")
    if "step" in df.columns:
        print(f"Minimum step: {df['step'].min()}")
        print(f"Maximum step: {df['step'].max()}")
        print(f"Unique steps: {df['step'].nunique():,}")

        print("\nFraud counts by step — first 10 steps:")
        print(
            df.groupby("step")["isFraud"]
            .agg(
                transactions="count",
                frauds="sum",
            )
            .head(10)
            .to_string()
            if "isFraud" in df.columns
            else df.groupby("step").size().head(10).to_string()
        )

    print("\n=== DUPLICATE ROWS ===")
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")


if __name__ == "__main__":
    main()
