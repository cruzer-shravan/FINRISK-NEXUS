import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SOURCE = Path("data/raw/paysim/PS_20174392719_1491204439457_log.csv")
OUT_DIR = Path("reports/data_exploration")
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Aggregate the entire dataset without keeping it in memory.
records = []

for chunk in pd.read_csv(
    SOURCE,
    usecols=["step", "isFraud", "isFlaggedFraud"],
    chunksize=250_000,
):
    summary = (
        chunk.groupby("step")
        .agg(
            transactions=("isFraud", "size"),
            fraud_count=("isFraud", "sum"),
            flagged_count=("isFlaggedFraud", "sum"),
        )
        .reset_index()
    )
    records.append(summary)

step_data = (
    pd.concat(records, ignore_index=True)
    .groupby("step", as_index=False)[["transactions", "fraud_count", "flagged_count"]]
    .sum()
    .sort_values("step")
)

step_data["fraud_rate_pct"] = 100 * step_data["fraud_count"] / step_data["transactions"]

step_data["flag_rate_pct"] = 100 * step_data["flagged_count"] / step_data["transactions"]

step_data["fraud_rolling_24_step_pct"] = (
    step_data["fraud_rate_pct"].rolling(window=24, min_periods=1).mean()
)

# Save step-level results.
csv_path = OUT_DIR / "paysim_temporal_summary.csv"
step_data.to_csv(csv_path, index=False)

# Overall statistics.
total_transactions = int(step_data["transactions"].sum())
total_fraud = int(step_data["fraud_count"].sum())
total_flagged = int(step_data["flagged_count"].sum())

overall_fraud_rate = 100 * total_fraud / total_transactions

peak_volume = step_data.loc[step_data["transactions"].idxmax()]
peak_fraud_count = step_data.loc[step_data["fraud_count"].idxmax()]
peak_fraud_rate = step_data.loc[step_data["fraud_rate_pct"].idxmax()]

# Avoid ranking tiny-volume steps as meaningful fraud hotspots.
eligible = step_data.loc[step_data["transactions"] >= 1000]
top_fraud_rates = eligible.nlargest(10, "fraud_rate_pct")

# Plot transaction volume and fraud counts.
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(
    step_data["step"],
    step_data["transactions"],
    label="Transaction volume",
)
ax1.set_xlabel("Simulation step")
ax1.set_ylabel("Transactions")

ax2 = ax1.twinx()
ax2.plot(
    step_data["step"],
    step_data["fraud_count"],
    label="Fraud-labelled transactions",
)
ax2.set_ylabel("Fraud count")

fig.suptitle("PaySim Transaction Volume and Fraud Counts")
fig.tight_layout()
fig.savefig(
    FIG_DIR / "paysim_temporal_volume_fraud.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)

# Plot fraud rate and smoothed trend.
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    step_data["step"],
    step_data["fraud_rate_pct"],
    alpha=0.45,
    label="Fraud rate per step",
)
ax.plot(
    step_data["step"],
    step_data["fraud_rolling_24_step_pct"],
    label="24-step rolling mean of step rates",
)
ax.set_xlabel("Simulation step")
ax.set_ylabel("Fraud rate (%)")
ax.set_title("PaySim Fraud Rate Over Time")
ax.legend()
fig.tight_layout()
fig.savefig(
    FIG_DIR / "paysim_temporal_fraud_rate.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)

report = {
    "total_transactions": total_transactions,
    "total_fraud": total_fraud,
    "overall_fraud_rate_pct": overall_fraud_rate,
    "total_flagged": total_flagged,
    "number_of_observed_steps": int(step_data["step"].nunique()),
    "minimum_step": int(step_data["step"].min()),
    "maximum_step": int(step_data["step"].max()),
    "peak_volume_step": int(peak_volume["step"]),
    "peak_volume_transactions": int(peak_volume["transactions"]),
    "peak_fraud_count_step": int(peak_fraud_count["step"]),
    "peak_fraud_count": int(peak_fraud_count["fraud_count"]),
    "peak_fraud_rate_step": int(peak_fraud_rate["step"]),
    "peak_fraud_rate_pct": float(peak_fraud_rate["fraud_rate_pct"]),
    "top_fraud_rate_steps_min_1000_transactions": (
        top_fraud_rates[["step", "transactions", "fraud_count", "fraud_rate_pct"]].to_dict(
            orient="records"
        )
    ),
}

json_path = OUT_DIR / "paysim_temporal_report.json"
json_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("=" * 60)
print("PAYSIM TEMPORAL EXPLORATION")
print("=" * 60)
print(f"Transactions: {total_transactions:,}")
print(f"Fraud-labelled: {total_fraud:,}")
print(f"Overall fraud rate: {overall_fraud_rate:.6f}%")
print(f"Rule-flagged transactions: {total_flagged:,}")
print(
    f"Observed steps: {report['minimum_step']} to "
    f"{report['maximum_step']} "
    f"({report['number_of_observed_steps']} distinct steps)"
)
print(
    f"Peak transaction volume: step {report['peak_volume_step']} "
    f"({report['peak_volume_transactions']:,} transactions)"
)
print(
    f"Peak fraud count: step {report['peak_fraud_count_step']} "
    f"({report['peak_fraud_count']:,} fraud-labelled transactions)"
)
print(
    f"Peak fraud rate: step {report['peak_fraud_rate_step']} ({report['peak_fraud_rate_pct']:.4f}%)"
)

print("\nTop fraud-rate steps with at least 1,000 transactions:")
print(
    top_fraud_rates[["step", "transactions", "fraud_count", "fraud_rate_pct"]].to_string(
        index=False
    )
)

print(f"\nCSV saved: {csv_path}")
print(f"JSON saved: {json_path}")
print(f"Plots saved in: {FIG_DIR}")
