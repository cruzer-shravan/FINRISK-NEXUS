from pathlib import Path

import pandas as pd

BASE = Path("data/raw/home_credit")
CHUNK_SIZE = 100_000

CHECKS = [
    {
        "child_file": "bureau_balance.csv",
        "child_key": "SK_ID_BUREAU",
        "parent_file": "bureau.csv",
        "parent_key": "SK_ID_BUREAU",
        "samples": [5001709, 5001740, 5001761, 5001781, 5001801],
    },
    {
        "child_file": "POS_CASH_balance.csv",
        "child_key": "SK_ID_PREV",
        "parent_file": "previous_application.csv",
        "parent_key": "SK_ID_PREV",
        "samples": [1000021, 1000037, 1000041, 1000095, 1000104],
    },
    {
        "child_file": "credit_card_balance.csv",
        "child_key": "SK_ID_PREV",
        "parent_file": "previous_application.csv",
        "parent_key": "SK_ID_PREV",
        "samples": [1000096, 1000344, 1000455, 1000514, 1000515],
    },
    {
        "child_file": "installments_payments.csv",
        "child_key": "SK_ID_PREV",
        "parent_file": "previous_application.csv",
        "parent_key": "SK_ID_PREV",
        "samples": [1000021, 1000037, 1000041, 1000095, 1000096],
    },
]


def read_key_values(filename, key):
    values = set()

    for chunk in pd.read_csv(
        BASE / filename,
        usecols=[key],
        chunksize=CHUNK_SIZE,
    ):
        values.update(chunk[key].dropna().tolist())

    return values


def main():
    for check in CHECKS:
        child_file = check["child_file"]
        child_key = check["child_key"]
        parent_file = check["parent_file"]
        parent_key = check["parent_key"]

        print(f"\n{'=' * 65}")
        print(f"Child:  {child_file} [{child_key}]")
        print(f"Parent: {parent_file} [{parent_key}]")

        child_ids = read_key_values(child_file, child_key)
        parent_ids = read_key_values(parent_file, parent_key)

        print(f"Distinct child IDs:  {len(child_ids):,}")
        print(f"Distinct parent IDs: {len(parent_ids):,}")

        for sample in check["samples"]:
            print(
                f"ID {sample}: "
                f"child_present={sample in child_ids}, "
                f"parent_present={sample in parent_ids}"
            )


if __name__ == "__main__":
    main()
