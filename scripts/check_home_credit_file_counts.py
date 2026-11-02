from pathlib import Path

import pandas as pd

BASE = Path("data/raw/home_credit")

FILES = [
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
]

EXPECTED_ROWS = {
    "bureau.csv": 1_716_428,
    "bureau_balance.csv": 27_299_925,
    "previous_application.csv": 1_670_214,
    "POS_CASH_balance.csv": 10_001_358,
    "credit_card_balance.csv": 3_840_312,
    "installments_payments.csv": 13_605_401,
}


def main():
    for filename in FILES:
        path = BASE / filename

        if not path.exists():
            print(f"MISSING: {filename}")
            continue

        # Count data rows without loading the full dataset into memory.
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            row_count = sum(1 for _ in file) - 1

        # Read only the header to determine the column count.
        columns = pd.read_csv(path, nrows=0).columns.tolist()

        expected = EXPECTED_ROWS[filename]
        status = "MATCH" if row_count == expected else "MISMATCH"

        print(f"\nFile: {filename}")
        print(f"  Rows: {row_count:,}")
        print(f"  Expected: {expected:,}")
        print(f"  Columns: {len(columns)}")
        print(f"  Status: {status}")
        print(f"  Headers: {columns}")


if __name__ == "__main__":
    main()
