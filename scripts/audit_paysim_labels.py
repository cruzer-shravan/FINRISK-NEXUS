import json
from pathlib import Path

import pandas as pd

SOURCE = Path("data/raw/paysim/PS_20174392719_1491204439457_log.csv")
OUTPUT_DIR = Path("reports/data_exploration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Counts are accumulated in chunks to limit memory usage.
label_counts = {}
type_counts = {}
total_rows = 0

for chunk in pd.read_csv(SOURCE, chunksize=250_000):
    total_rows += len(chunk)

    # Actual fraud label versus existing rule-based flag
    cross_tab = pd.crosstab(
        chunk["isFraud"],
        chunk["isFlaggedFraud"],
    )

    for fraud_label in [0, 1]:
        for flag_label in [0, 1]:
            count = int(cross_tab.get(flag_label, pd.Series(dtype=int)).get(fraud_label, 0))
            key = f"fraud_{fraud_label}_flag_{flag_label}"
            label_counts[key] = label_counts.get(key, 0) + count

    # Transaction volume and fraud counts by type
    grouped = chunk.groupby(["type", "isFraud"]).size()

    for (transaction_type, fraud_label), count in grouped.items():
        key = f"{transaction_type}|{int(fraud_label)}"
        type_counts[key] = type_counts.get(key, 0) + int(count)

# Interpret the four label/flag combinations
tn = label_counts.get("fraud_0_flag_0", 0)
fp = label_counts.get("fraud_0_flag_1", 0)
fn = label_counts.get("fraud_1_flag_0", 0)
tp = label_counts.get("fraud_1_flag_1", 0)

flagged_total = tp + fp
fraud_total = tp + fn

results = {
    "total_rows": total_rows,
    "label_flag_combinations": {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    },
    "rule_flagged_transactions": flagged_total,
    "fraud_label_count": fraud_total,
    "rule_precision": tp / flagged_total if flagged_total else None,
    "rule_recall": tp / fraud_total if fraud_total else None,
    "fraud_by_transaction_type": {},
}

for transaction_type in sorted({key.split("|")[0] for key in type_counts}):
    legitimate = type_counts.get(f"{transaction_type}|0", 0)
    fraud = type_counts.get(f"{transaction_type}|1", 0)
    total = legitimate + fraud

    results["fraud_by_transaction_type"][transaction_type] = {
        "total_transactions": total,
        "legitimate_label_count": legitimate,
        "fraud_label_count": fraud,
        "fraud_rate_percent": (100 * fraud / total if total else 0),
    }

output_file = OUTPUT_DIR / "paysim_label_audit.json"
output_file.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print("=" * 55)
print("PAYSIM LABEL AUDIT")
print("=" * 55)
print(f"Total transactions: {total_rows:,}")
print("\nActual fraud label vs rule-based flag")
print(f"True negatives : {tn:,}")
print(f"False positives: {fp:,}")
print(f"False negatives: {fn:,}")
print(f"True positives : {tp:,}")
print(f"Rule precision : {results['rule_precision']}")
print(f"Rule recall    : {results['rule_recall']}")

print("\nFraud by transaction type")
for name, stats in results["fraud_by_transaction_type"].items():
    print(
        f"{name:10} | "
        f"Transactions: {stats['total_transactions']:>10,} | "
        f"Fraud: {stats['fraud_label_count']:>6,} | "
        f"Rate: {stats['fraud_rate_percent']:.4f}%"
    )

print(f"\nReport saved: {output_file}")
