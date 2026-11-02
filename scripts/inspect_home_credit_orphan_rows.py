from pathlib import Path

import pandas as pd

BASE = Path("data/raw/home_credit")
CHUNK_SIZE = 100_000

CHECKS = [
    {
        "child": "bureau_balance.csv",
        "child_key": "SK_ID_BUREAU",
        "parent": "bureau.csv",
        "parent_key": "SK_ID_BUREAU",
        "samples": [5001709, 5001740, 5001761],
    },
    {
        "child": "POS_CASH_balance.csv",
        "child_key": "SK_ID_PREV",
        "parent": "previous_application.csv",
        "parent_key": "SK_ID_PREV",
        "samples": [1000021, 1000037, 1000041],
    },
    {
        "child": "credit_card_balance.csv",
        "child_key": "SK_ID_PREV",
        "parent": "previous_application.csv",
        "parent_key": "SK_ID_PREV",
        "samples": [1000096, 1000344, 1000455],
    },
    {
        "child": "installments_payments.csv",
        "child_key": "SK_ID_PREV",
        "parent": "previous_application.csv",
        "parent_key": "SK_ID_PREV",
        "samples": [1000021, 1000037, 1000096],
    },
]


def find_rows(filename, key, sample_ids):
    path = BASE / filename
    matches = []

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        found = chunk[chunk[key].isin(sample_ids)]
        if not found.empty:
            matches.append(found)

    if matches:
        return pd.concat(matches, ignore_index=True)

    return pd.DataFrame()


def main():
    for check in CHECKS:
        print("\n" + "=" * 70)
        print(f"Child file:  {check['child']}")
        print(f"Parent file: {check['parent']}")

        child_rows = find_rows(
            check["child"],
            check["child_key"],
            check["samples"],
        )

        parent_rows = find_rows(
            check["parent"],
            check["parent_key"],
            check["samples"],
        )

        print("\nMatching child rows:")
        print(
            child_rows.head(10).to_string(index=False) if not child_rows.empty else "No rows found"
        )

        print("\nMatching parent rows:")
        print(
            parent_rows.head(10).to_string(index=False)
            if not parent_rows.empty
            else "No rows found"
        )


if __name__ == "__main__":
    main()
