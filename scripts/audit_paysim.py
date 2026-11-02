import json
from pathlib import Path

import pandas as pd

source = Path("data/raw/paysim/PS_20174392719_1491204439457_log.csv")
output = Path("reports/data_exploration")
output.mkdir(parents=True, exist_ok=True)

total_rows = 0
missing = None
fraud_counts = {0: 0, 1: 0}
flag_counts = {0: 0, 1: 0}
fraud_by_type = {}
duplicate_rows = 0
invalid_amount_rows = 0
negative_amount_rows = 0
invalid_label_rows = 0
minimum_step = None
maximum_step = None
minimum_amount = None
maximum_amount = None
transaction_types = set()
dtypes = None
expected_columns = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

for chunk in pd.read_csv(source, chunksize=250_000):
    total_rows += len(chunk)

    if dtypes is None:
        dtypes = {k: str(v) for k, v in chunk.dtypes.items()}
        missing = pd.Series(0, index=chunk.columns, dtype="int64")

    missing += chunk.isna().sum()
    duplicate_rows += int(chunk.duplicated().sum())

    invalid_amount_rows += int(
        (~chunk["amount"].map(lambda x: pd.notna(x) and x < float("inf"))).sum()
    )
    negative_amount_rows += int((chunk["amount"] < 0).sum())

    invalid_label_rows += int(
        (~chunk["isFraud"].isin([0, 1])).sum() + (~chunk["isFlaggedFraud"].isin([0, 1])).sum()
    )

    for col, counts in [
        ("isFraud", fraud_counts),
        ("isFlaggedFraud", flag_counts),
    ]:
        values = chunk[col].value_counts()
        for key in counts:
            counts[key] += int(values.get(key, 0))

    by_type = chunk.groupby(["type", "isFraud"]).size()
    for (kind, label), count in by_type.items():
        fraud_by_type[f"{kind}|{label}"] = fraud_by_type.get(f"{kind}|{label}", 0) + int(count)

    transaction_types.update(chunk["type"].dropna().unique().tolist())

    lo, hi = chunk["step"].min(), chunk["step"].max()
    minimum_step = lo if minimum_step is None else min(minimum_step, lo)
    maximum_step = hi if maximum_step is None else max(maximum_step, hi)

    lo, hi = chunk["amount"].min(), chunk["amount"].max()
    minimum_amount = lo if minimum_amount is None else min(minimum_amount, lo)
    maximum_amount = hi if maximum_amount is None else max(maximum_amount, hi)

report = {
    "source_file": str(source),
    "rows": total_rows,
    "columns": expected_columns,
    "column_count": len(expected_columns),
    "columns_match_expected": list(dtypes) == expected_columns,
    "dtypes": dtypes,
    "missing_values": {k: int(v) for k, v in missing.items()},
    "duplicate_rows_within_chunks": duplicate_rows,
    "fraud_counts": {str(k): v for k, v in fraud_counts.items()},
    "fraud_rate_percent": round(100 * fraud_counts[1] / total_rows, 6) if total_rows else None,
    "flag_counts": {str(k): v for k, v in flag_counts.items()},
    "fraud_by_transaction_type_and_label": fraud_by_type,
    "transaction_types": sorted(transaction_types),
    "invalid_amount_rows": invalid_amount_rows,
    "negative_amount_rows": negative_amount_rows,
    "invalid_label_rows": invalid_label_rows,
    "min_step": int(minimum_step),
    "max_step": int(maximum_step),
    "min_amount": float(minimum_amount),
    "max_amount": float(maximum_amount),
}

report_path = output / "paysim_data_quality_report.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("PAYSim DATA-QUALITY AUDIT COMPLETED")
print("=" * 45)
print("Rows:", f"{total_rows:,}")
print("Columns:", len(expected_columns))
print("Column names match expected:", report["columns_match_expected"])
print("Missing values:", sum(report["missing_values"].values()))
print("Duplicate rows within chunks:", duplicate_rows)
print("Fraud counts:", fraud_counts)
print("Fraud rate (%):", report["fraud_rate_percent"])
print("Flag counts:", flag_counts)
print("Transaction types:", sorted(transaction_types))
print("Invalid amount rows:", invalid_amount_rows)
print("Negative amount rows:", negative_amount_rows)
print("Invalid label rows:", invalid_label_rows)
print("Step range:", minimum_step, "to", maximum_step)
print("Amount range:", minimum_amount, "to", maximum_amount)
print("Report saved:", report_path)
