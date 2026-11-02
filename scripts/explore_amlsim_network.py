from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("data/raw/ibm_amlsim/1Kvertices-100Kedges")
OUT = Path("reports/ibm_amlsim")
FIGURES = OUT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

accounts = pd.read_csv(ROOT / "accounts.csv")
tx = pd.read_csv(ROOT / "transactions.csv")
alerts = pd.read_csv(ROOT / "alerts.csv")

# Normalize label types for analysis.
tx["IS_FRAUD"] = tx["IS_FRAUD"].astype(bool)
accounts["IS_FRAUD"] = accounts["IS_FRAUD"].astype(bool)

summary = []


def record(title, value):
    line = f"{title}: {value}"
    summary.append(line)
    print(line)


print("\n========== 1. TRANSACTION BEHAVIOR ==========")

record("Total transactions", f"{len(tx):,}")
record("Unique senders", tx["SENDER_ACCOUNT_ID"].nunique())
record("Unique receivers", tx["RECEIVER_ACCOUNT_ID"].nunique())
record(
    "Unique participating accounts",
    len(set(tx["SENDER_ACCOUNT_ID"]).union(tx["RECEIVER_ACCOUNT_ID"])),
)
record("Fraud-labelled transactions", int(tx["IS_FRAUD"].sum()))
record("Fraud transaction rate", f"{tx['IS_FRAUD'].mean():.4%}")

record("Zero-amount transactions", int((tx["TX_AMOUNT"] == 0).sum()))
record("Negative-amount transactions", int((tx["TX_AMOUNT"] < 0).sum()))
record("Negative timestamps", int((tx["TIMESTAMP"] < 0).sum()))

print("\nTransaction amount summary:")
print(tx["TX_AMOUNT"].describe().to_string())

print("\nTransactions per timestamp:")
print(tx.groupby("TIMESTAMP").size().describe().to_string())

# Activity over simulation time steps.
time_activity = (
    tx.groupby("TIMESTAMP")
    .agg(
        transaction_count=("TX_ID", "size"),
        total_amount=("TX_AMOUNT", "sum"),
        fraud_count=("IS_FRAUD", "sum"),
    )
    .reset_index()
)

time_activity.to_csv(OUT / "transaction_time_activity.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    time_activity["TIMESTAMP"],
    time_activity["transaction_count"],
)
ax.set(
    title="Transaction Volume Over Simulation Time",
    xlabel="Simulation timestamp",
    ylabel="Transaction count",
)
fig.tight_layout()
fig.savefig(FIGURES / "transaction_volume_over_time.png", dpi=150)
plt.close(fig)


print("\n========== 2. ACCOUNT NETWORK ==========")

# Each transaction is a directed edge:
# sender account -> receiver account.
outgoing = tx.groupby("SENDER_ACCOUNT_ID").agg(
    outgoing_count=("TX_ID", "size"),
    outgoing_amount=("TX_AMOUNT", "sum"),
    outgoing_fraud_count=("IS_FRAUD", "sum"),
)

incoming = tx.groupby("RECEIVER_ACCOUNT_ID").agg(
    incoming_count=("TX_ID", "size"),
    incoming_amount=("TX_AMOUNT", "sum"),
    incoming_fraud_count=("IS_FRAUD", "sum"),
)

network = accounts.set_index("ACCOUNT_ID").join(outgoing, how="left").join(incoming, how="left")

count_cols = [
    "outgoing_count",
    "incoming_count",
    "outgoing_fraud_count",
    "incoming_fraud_count",
]
amount_cols = ["outgoing_amount", "incoming_amount"]

network[count_cols] = network[count_cols].fillna(0)
network[amount_cols] = network[amount_cols].fillna(0)

network[count_cols] = network[count_cols].astype(int)

network["total_transactions"] = network["outgoing_count"] + network["incoming_count"]
network["total_amount_moved"] = network["outgoing_amount"] + network["incoming_amount"]
network["unique_counterparties"] = (
    tx.groupby("SENDER_ACCOUNT_ID")["RECEIVER_ACCOUNT_ID"]
    .nunique()
    .reindex(network.index)
    .fillna(0)
    .add(
        tx.groupby("RECEIVER_ACCOUNT_ID")["SENDER_ACCOUNT_ID"]
        .nunique()
        .reindex(network.index)
        .fillna(0),
        fill_value=0,
    )
)

# The counterparty sum above may count the same account in both
# directions. We will calculate exact distinct counterparties below.
counterparties = {account_id: set() for account_id in accounts["ACCOUNT_ID"]}

for sender, receiver in zip(tx["SENDER_ACCOUNT_ID"], tx["RECEIVER_ACCOUNT_ID"], strict=True):
    counterparties[sender].add(receiver)
    counterparties[receiver].add(sender)

network["unique_counterparties"] = [len(counterparties[account_id]) for account_id in network.index]

network["account_label"] = network["IS_FRAUD"].astype(bool)
network["fraud_tx_involvement"] = network["outgoing_fraud_count"] + network["incoming_fraud_count"]

network.to_csv(OUT / "account_network_metrics.csv")

record("Accounts in network", len(network))
record("Accounts with outgoing transactions", int((network["outgoing_count"] > 0).sum()))
record("Accounts with incoming transactions", int((network["incoming_count"] > 0).sum()))
record("Accounts with no transaction activity", int((network["total_transactions"] == 0).sum()))
record(
    "Accounts involved in fraud-labelled transactions",
    int((network["fraud_tx_involvement"] > 0).sum()),
)
record("Fraud-labelled accounts", int(network["account_label"].sum()))

print("\nTop 10 accounts by total transaction count:")
print(
    network.sort_values("total_transactions", ascending=False)[
        [
            "outgoing_count",
            "incoming_count",
            "unique_counterparties",
            "account_label",
            "fraud_tx_involvement",
        ]
    ]
    .head(10)
    .to_string()
)

print("\nTop 10 accounts by total amount moved:")
print(
    network.sort_values("total_amount_moved", ascending=False)[
        [
            "outgoing_amount",
            "incoming_amount",
            "total_amount_moved",
            "account_label",
            "fraud_tx_involvement",
        ]
    ]
    .head(10)
    .to_string()
)

print("\nAccount label vs transaction involvement:")
print(
    pd.crosstab(
        network["account_label"],
        network["fraud_tx_involvement"] > 0,
        rownames=["Account labelled fraud"],
        colnames=["Involved in fraud-labelled transaction"],
    ).to_string()
)


print("\n========== 3. NETWORK CONCENTRATION ==========")

# Unique directed sender-receiver pairs and repeated transfers.
edges = (
    tx.groupby(["SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID"])
    .agg(
        transaction_count=("TX_ID", "size"),
        total_amount=("TX_AMOUNT", "sum"),
        fraud_count=("IS_FRAUD", "sum"),
    )
    .reset_index()
)

record("Unique directed account pairs", len(edges))
record("Repeated-transfer pairs", int((edges["transaction_count"] > 1).sum()))
record(
    "Self-transfer transactions", int((tx["SENDER_ACCOUNT_ID"] == tx["RECEIVER_ACCOUNT_ID"]).sum())
)

print("\nTop 10 directed account pairs by transfer count:")
print(edges.sort_values("transaction_count", ascending=False).head(10).to_string(index=False))

edges.to_csv(OUT / "account_network_edges.csv", index=False)

# Degree here means number of distinct counterparties, not
# graph-theoretic centrality.
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(network["unique_counterparties"], bins=30)
ax.set(
    title="Distribution of Account Counterparties",
    xlabel="Distinct counterparties",
    ylabel="Number of accounts",
)
fig.tight_layout()
fig.savefig(FIGURES / "counterparty_distribution.png", dpi=150)
plt.close(fig)


print("\n========== 4. FRAUD AND ALERT ANALYSIS ==========")

fraud_tx = tx[tx["IS_FRAUD"]].copy()

print("\nFraud-labelled transaction amount summary:")
print(fraud_tx["TX_AMOUNT"].describe().to_string())

print("\nFraud-labelled transactions over time:")
print(fraud_tx.groupby("TIMESTAMP").size().sort_values(ascending=False).head(10).to_string())

print("\nAlert type distribution:")
print(alerts["ALERT_TYPE"].value_counts().to_string())

print("\nAlert ID frequency:")
print(alerts["ALERT_ID"].value_counts().describe().to_string())

# Alert IDs repeat, so summarize both alert records and unique IDs.
alert_groups = (
    alerts.groupby("ALERT_ID")
    .agg(
        alert_records=("TX_ID", "size"),
        unique_transactions=("TX_ID", "nunique"),
        unique_senders=("SENDER_ACCOUNT_ID", "nunique"),
        unique_receivers=("RECEIVER_ACCOUNT_ID", "nunique"),
        total_amount=("TX_AMOUNT", "sum"),
        alert_types=("ALERT_TYPE", "nunique"),
    )
    .reset_index()
)

alert_groups.to_csv(OUT / "alert_group_summary.csv", index=False)

print("\nAlert group summary:")
print(alert_groups.describe(include="all").to_string())

# Save the run's summary.
(OUT / "exploration_summary.txt").write_text(
    "\n".join(summary) + "\n",
    encoding="utf-8",
)

print(f"\nExploration complete. Outputs saved to: {OUT}")
