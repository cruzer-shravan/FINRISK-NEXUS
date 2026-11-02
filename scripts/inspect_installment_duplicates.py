from pathlib import Path

import pandas as pd

path = Path("data/raw/home_credit/installments_payments.csv")

keys = [
    "SK_ID_PREV",
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER",
]

df = pd.read_csv(
    path,
    usecols=keys
    + [
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ],
)

duplicate_mask = df.duplicated(subset=keys, keep=False)
duplicates = df.loc[duplicate_mask].sort_values(keys)

print("Total rows loaded:", len(df))
print("Rows belonging to duplicate key groups:", len(duplicates))
print("Duplicate key groups:", duplicates[keys].drop_duplicates().shape[0])

print("\nFirst 30 rows from duplicate groups:")
print(duplicates.head(30).to_string(index=False))
