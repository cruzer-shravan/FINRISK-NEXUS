from pathlib import Path

import pandas as pd

ROOT = Path("data/raw/ibm_amlsim/1Kvertices-100Kedges")

FILES = ["accounts.csv", "transactions.csv", "alerts.csv"]

for filename in FILES:
    path = ROOT / filename
    df = pd.read_csv(path)

    print(f"\n{'=' * 70}")
    print(f"FILE: {filename}")
    print(f"{'=' * 70}")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nInferred data types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    print("\nUnique values per column:")
    print(df.nunique(dropna=False).to_string())

    print("\nSample records:")
    print(df.head(3).to_string(index=False))
