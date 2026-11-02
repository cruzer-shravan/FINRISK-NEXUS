import json
from pathlib import Path

import pandas as pd

SOURCE = Path("data/raw/paysim/PS_20174392719_1491204439457_log.csv")
OUT_DIR = Path("reports/data_exploration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

amount_stats = {}
type_stats = {}
balance_stats = {}
zero_amount_count = 0

for chunk in pd.read_csv(SOURCE, chunksize=250_000):
    zero_amount_count += int((chunk["amount"] == 0).sum())

    # Amount statistics by fraud label
    for label, group in chunk.groupby("isFraud"):
        stats = amount_stats.setdefault(
            str(label),
            {
                "count": 0,
                "sum": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
                "zero_amount_count": 0,
            },
        )

        stats["count"] += len(group)
        stats["sum"] += group["amount"].sum()
        stats["min"] = min(stats["min"], group["amount"].min())
        stats["max"] = max(stats["max"], group["amount"].max())
        stats["zero_amount_count"] += int((group["amount"] == 0).sum())

    # Amount and fraud counts by transaction type
    grouped = chunk.groupby(["type", "isFraud"]).agg(
        transactions=("amount", "size"),
        total_amount=("amount", "sum"),
        mean_amount=("amount", "mean"),
    )

    for (kind, fraud), row in grouped.iterrows():
        key = f"{kind}|{int(fraud)}"
        stats = type_stats.setdefault(
            key,
            {
                "transactions": 0,
                "total_amount": 0.0,
                "weighted_amount_sum": 0.0,
            },
        )
        stats["transactions"] += int(row["transactions"])
        stats["total_amount"] += float(row["total_amount"])
        stats["weighted_amount_sum"] += float(row["mean_amount"]) * int(row["transactions"])

    # Diagnostic balance checks; mismatches are not
    # automatically data errors in this simulation.
    chunk["origin_balance_residual"] = (
        chunk["oldbalanceOrg"] - chunk["amount"] - chunk["newbalanceOrig"]
    )
    chunk["destination_balance_residual"] = (
        chunk["oldbalanceDest"] + chunk["amount"] - chunk["newbalanceDest"]
    )

    for kind, group in chunk.groupby("type"):
        stats = balance_stats.setdefault(
            kind,
            {
                "transactions": 0,
                "origin_residual_mismatch": 0,
                "destination_residual_mismatch": 0,
            },
        )
        stats["transactions"] += len(group)
        stats["origin_residual_mismatch"] += int(
            (group["origin_balance_residual"].abs() > 0.01).sum()
        )
        stats["destination_residual_mismatch"] += int(
            (group["destination_balance_residual"].abs() > 0.01).sum()
        )

# Finalize statistics
for stats in amount_stats.values():
    stats["mean_amount"] = stats["sum"] / stats["count"]
    stats.pop("sum")

for stats in type_stats.values():
    stats["mean_amount"] = stats.pop("weighted_amount_sum") / stats["transactions"]

report = {
    "zero_amount_transactions": zero_amount_count,
    "amount_by_fraud_label": amount_stats,
    "amount_by_transaction_type_and_fraud": type_stats,
    "balance_residual_diagnostics_by_type": balance_stats,
}

output_file = OUT_DIR / "paysim_amount_balance_report.json"
output_file.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("=" * 60)
print("PAYSIM AMOUNT AND BALANCE EXPLORATION")
print("=" * 60)

print("\nAmount statistics by fraud label")
for label, stats in amount_stats.items():
    print(
        f"Label={label} | Count={stats['count']:,} | "
        f"Mean={stats['mean_amount']:,.2f} | "
        f"Min={stats['min']:,.2f} | "
        f"Max={stats['max']:,.2f} | "
        f"Zero={stats['zero_amount_count']:,}"
    )

print(f"\nZero-amount transactions overall: {zero_amount_count:,}")

print("\nAmount statistics by transaction type and fraud label")
for key, stats in sorted(type_stats.items()):
    print(f"{key:20} | Count={stats['transactions']:>10,} | Mean={stats['mean_amount']:>14,.2f}")

print("\nBalance residual mismatches by transaction type")
for kind, stats in sorted(balance_stats.items()):
    print(
        f"{kind:10} | Count={stats['transactions']:>10,} | "
        f"Origin mismatches="
        f"{stats['origin_residual_mismatch']:,} | "
        f"Destination mismatches="
        f"{stats['destination_residual_mismatch']:,}"
    )

print(f"\nReport saved: {output_file}")
