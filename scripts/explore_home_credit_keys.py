from pathlib import Path

import pandas as pd

BASE = Path("data/raw/home_credit")

TABLES = {
    "application_train.csv": ["SK_ID_CURR"],
    "application_test.csv": ["SK_ID_CURR"],
    "bureau.csv": ["SK_ID_CURR", "SK_ID_BUREAU"],
    "bureau_balance.csv": ["SK_ID_BUREAU", "MONTHS_BALANCE"],
    "previous_application.csv": ["SK_ID_CURR", "SK_ID_PREV"],
    "POS_CASH_balance.csv": ["SK_ID_PREV", "MONTHS_BALANCE"],
    "credit_card_balance.csv": ["SK_ID_PREV", "MONTHS_BALANCE"],
    "installments_payments.csv": [
        "SK_ID_PREV",
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
    ],
}

SAMPLE_ROWS = 50_000

for filename, keys in TABLES.items():
    path = BASE / filename
    df = pd.read_csv(path, usecols=keys, nrows=SAMPLE_ROWS)

    print(f"\n{'=' * 60}")
    print(f"TABLE: {filename}")
    print(f"Rows inspected: {len(df):,}")
    print(f"Key columns: {keys}")

    for key in keys:
        print(
            f"{key}: "
            f"{df[key].nunique(dropna=True):,} distinct values; "
            f"{df[key].isna().sum():,} missing"
        )

    print(
        "Duplicate combinations of listed keys:",
        f"{df.duplicated(subset=keys).sum():,}",
    )
