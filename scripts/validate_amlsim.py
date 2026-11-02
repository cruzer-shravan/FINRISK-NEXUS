from pathlib import Path

import pandas as pd

ROOT = Path("data/raw/ibm_amlsim/1Kvertices-100Kedges")

accounts = pd.read_csv(ROOT / "accounts.csv")
transactions = pd.read_csv(ROOT / "transactions.csv")
alerts = pd.read_csv(ROOT / "alerts.csv")


def section(title):
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")


# 1. Dataset sizes and account-level labels
section("DATASET SIZES")
for name, df in [
    ("Accounts", accounts),
    ("Transactions", transactions),
    ("Alerts", alerts),
]:
    print(f"{name}: {len(df):,} rows")

section("ACCOUNT ANALYSIS")
print("Unique account IDs:", accounts["ACCOUNT_ID"].nunique())
print("Duplicate account IDs:", accounts["ACCOUNT_ID"].duplicated().sum())
print("\nAccount fraud labels:")
print(accounts["IS_FRAUD"].value_counts(dropna=False).to_string())

# 2. Transaction-level labels and values
section("TRANSACTION ANALYSIS")
print("\nFraud label distribution:")
print(transactions["IS_FRAUD"].value_counts(dropna=False).to_string())
print("\nTransaction type distribution:")
print(transactions["TX_TYPE"].value_counts(dropna=False).to_string())
print("\nTimestamp range:")
print(transactions["TIMESTAMP"].agg(["min", "max"]).to_string())
print("\nTransaction amount summary:")
print(transactions["TX_AMOUNT"].describe().to_string())

print("\nDuplicate transaction IDs:", transactions["TX_ID"].duplicated().sum())

# 3. Alert-level analysis
section("ALERT ANALYSIS")
print("Unique alert IDs:", alerts["ALERT_ID"].nunique())
print("\nRows per alert ID:")
print(alerts["ALERT_ID"].value_counts().describe().to_string())

print("\nAlert types:")
print(alerts["ALERT_TYPE"].value_counts(dropna=False).to_string())

print("\nAlert fraud labels:")
print(alerts["IS_FRAUD"].value_counts(dropna=False).to_string())

# 4. Referential integrity
account_ids = set(accounts["ACCOUNT_ID"])
tx_ids = set(transactions["TX_ID"])
alert_ids = set(alerts["ALERT_ID"])

section("RELATIONSHIP VALIDATION")

for col in ["SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID"]:
    unknown = ~transactions[col].isin(account_ids)
    print(f"Transactions with unknown {col}: {unknown.sum():,}")

print("Transactions whose TX_ID is absent from alerts:", (~alerts["TX_ID"].isin(tx_ids)).sum())

print("Alert rows whose TX_ID is absent from transactions:", (~alerts["TX_ID"].isin(tx_ids)).sum())

print("Transactions with ALERT_ID = -1:", (transactions["ALERT_ID"] == -1).sum())

print("Transactions with ALERT_ID found in alerts:", transactions["ALERT_ID"].isin(alert_ids).sum())

print("Alert IDs appearing in multiple rows:", (alerts["ALERT_ID"].value_counts() > 1).sum())

# Compare transaction and alert fraud labels for matching TX_IDs
matched = alerts.merge(
    transactions[["TX_ID", "IS_FRAUD"]],
    on="TX_ID",
    how="inner",
    suffixes=("_alert", "_transaction"),
)

section("LABEL CONSISTENCY")
print("Alert rows matched to transactions:", len(matched))
if not matched.empty:
    print(
        "Fraud-label disagreements:",
        (matched["IS_FRAUD_alert"] != matched["IS_FRAUD_transaction"]).sum(),
    )

print("\nValidation complete. No source files were modified.")
