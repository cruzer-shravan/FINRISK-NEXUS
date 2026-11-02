from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path("data/raw/ibm_amlsim/1Kvertices-100Kedges")
OUT = Path("reports/ibm_amlsim")
OUT.mkdir(parents=True, exist_ok=True)

accounts = pd.read_csv(ROOT / "accounts.csv")
tx = pd.read_csv(ROOT / "transactions.csv")
alerts = pd.read_csv(ROOT / "alerts.csv")

# ---------------------------------------------------------
# 1. Build account-level network and transaction features
# ---------------------------------------------------------
G = nx.DiGraph()
G.add_nodes_from(accounts["ACCOUNT_ID"].tolist())

pair_features = (
    tx.groupby(["SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID"])
    .agg(
        pair_transaction_count=("TX_ID", "size"),
        pair_total_amount=("TX_AMOUNT", "sum"),
    )
    .reset_index()
)

for row in pair_features.itertuples(index=False):
    G.add_edge(
        row.SENDER_ACCOUNT_ID,
        row.RECEIVER_ACCOUNT_ID,
        transaction_count=int(row.pair_transaction_count),
        total_amount=float(row.pair_total_amount),
    )

node_features = pd.DataFrame(
    {
        "ACCOUNT_ID": list(G.nodes()),
        "in_degree": [G.in_degree(n) for n in G.nodes()],
        "out_degree": [G.out_degree(n) for n in G.nodes()],
    }
)

# Transaction-level volume and amount features.
outgoing = tx.groupby("SENDER_ACCOUNT_ID").agg(
    outgoing_tx_count=("TX_ID", "size"),
    outgoing_amount=("TX_AMOUNT", "sum"),
    outgoing_fraud_tx_count=("IS_FRAUD", "sum"),
)

incoming = tx.groupby("RECEIVER_ACCOUNT_ID").agg(
    incoming_tx_count=("TX_ID", "size"),
    incoming_amount=("TX_AMOUNT", "sum"),
    incoming_fraud_tx_count=("IS_FRAUD", "sum"),
)

outgoing.index.name = "ACCOUNT_ID"
incoming.index.name = "ACCOUNT_ID"

node_features = node_features.merge(outgoing, on="ACCOUNT_ID", how="left").merge(
    incoming, on="ACCOUNT_ID", how="left"
)

count_cols = [
    "outgoing_tx_count",
    "incoming_tx_count",
    "outgoing_fraud_tx_count",
    "incoming_fraud_tx_count",
]

amount_cols = ["outgoing_amount", "incoming_amount"]

node_features[count_cols] = node_features[count_cols].fillna(0).astype(int)
node_features[amount_cols] = node_features[amount_cols].fillna(0.0)

node_features["total_tx_count"] = (
    node_features["outgoing_tx_count"] + node_features["incoming_tx_count"]
)

node_features["total_amount_moved"] = (
    node_features["outgoing_amount"] + node_features["incoming_amount"]
)

node_features["total_degree"] = node_features["in_degree"] + node_features["out_degree"]

node_features["betweenness_centrality"] = pd.Series(nx.betweenness_centrality(G, normalized=True))

# Retrospective label for analysis only.
account_labels = accounts.set_index("ACCOUNT_ID")["IS_FRAUD"]
node_features["account_is_fraud"] = node_features["ACCOUNT_ID"].map(account_labels).astype(bool)

# Component membership.
strong_components = list(nx.strongly_connected_components(G))
component_lookup = {
    account_id: component_id
    for component_id, component in enumerate(strong_components)
    for account_id in component
}

node_features["strong_component_id"] = node_features["ACCOUNT_ID"].map(component_lookup)

node_features["strong_component_size"] = node_features["strong_component_id"].map(
    pd.Series({i: len(component) for i, component in enumerate(strong_components)})
)

# ---------------------------------------------------------
# 2. Fraud prevalence by feature ranking
# ---------------------------------------------------------
print("=" * 70)
print("FRAUD VALIDATION BY NETWORK RANK")
print("=" * 70)

ranking_cols = [
    "in_degree",
    "out_degree",
    "total_degree",
    "betweenness_centrality",
    "total_tx_count",
    "total_amount_moved",
]

rank_results = []

for feature in ranking_cols:
    ranked = node_features.sort_values(feature, ascending=False, kind="stable")

    for percentage in (1, 5, 10):
        n = max(1, int(len(ranked) * percentage / 100))
        subset = ranked.head(n)

        result = {
            "feature": feature,
            "top_percentage": percentage,
            "accounts_selected": n,
            "fraud_accounts": int(subset["account_is_fraud"].sum()),
            "fraud_prevalence_pct": (100 * subset["account_is_fraud"].mean()),
        }
        rank_results.append(result)

rank_df = pd.DataFrame(rank_results)
print(rank_df.to_string(index=False, float_format="%.3f"))

rank_df.to_csv(OUT / "network_fraud_rank_validation.csv", index=False)

# ---------------------------------------------------------
# 3. Compare fraud and non-fraud account characteristics
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("FRAUD VS NON-FRAUD ACCOUNT FEATURES")
print("=" * 70)

comparison_cols = [
    "in_degree",
    "out_degree",
    "total_degree",
    "betweenness_centrality",
    "incoming_tx_count",
    "outgoing_tx_count",
    "incoming_amount",
    "outgoing_amount",
    "total_tx_count",
    "total_amount_moved",
    "strong_component_size",
]

comparison = (
    node_features.groupby("account_is_fraud")[comparison_cols].agg(["median", "mean"]).round(4)
)

print(comparison.to_string())
comparison.to_csv(OUT / "fraud_network_feature_comparison.csv")

# ---------------------------------------------------------
# 4. Link alerts to accounts and network characteristics
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("ALERT TYPE VS NETWORK FEATURES")
print("=" * 70)

# Each alert row is associated with its sender and receiver.
alert_accounts = alerts.merge(
    node_features,
    left_on="SENDER_ACCOUNT_ID",
    right_on="ACCOUNT_ID",
    how="left",
    suffixes=("", "_sender"),
)

alert_accounts = alert_accounts.merge(
    node_features.add_prefix("receiver_"),
    left_on="RECEIVER_ACCOUNT_ID",
    right_on="receiver_ACCOUNT_ID",
    how="left",
)

alert_summary = (
    alert_accounts.groupby("ALERT_TYPE")
    .agg(
        alert_records=("ALERT_ID", "size"),
        unique_alerts=("ALERT_ID", "nunique"),
        unique_transactions=("TX_ID", "nunique"),
        unique_senders=("SENDER_ACCOUNT_ID", "nunique"),
        unique_receivers=("RECEIVER_ACCOUNT_ID", "nunique"),
        median_sender_in_degree=("in_degree", "median"),
        median_sender_out_degree=("out_degree", "median"),
        median_receiver_in_degree=("receiver_in_degree", "median"),
        median_receiver_out_degree=("receiver_out_degree", "median"),
        median_sender_betweenness=("betweenness_centrality", "median"),
        median_receiver_betweenness=("receiver_betweenness_centrality", "median"),
    )
    .round(4)
)

print(alert_summary.to_string())
alert_summary.to_csv(OUT / "alert_network_feature_comparison.csv")

# ---------------------------------------------------------
# 5. Alert account coverage and label consistency
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("ALERT COVERAGE")
print("=" * 70)

alert_senders = set(alerts["SENDER_ACCOUNT_ID"])
alert_receivers = set(alerts["RECEIVER_ACCOUNT_ID"])
alert_involved_accounts = alert_senders | alert_receivers

fraud_accounts = set(accounts.loc[accounts["IS_FRAUD"].astype(bool), "ACCOUNT_ID"])

print(f"Accounts in dataset: {len(accounts):,}")
print(f"Fraud-labelled accounts: {len(fraud_accounts):,}")
print(f"Accounts appearing in alerts: {len(alert_involved_accounts):,}")
print(
    "Fraud-labelled accounts appearing in alerts:",
    len(fraud_accounts & alert_involved_accounts),
)
print(
    "Fraud-labelled accounts absent from alerts:",
    len(fraud_accounts - alert_involved_accounts),
)

alert_coverage = pd.DataFrame(
    {
        "ACCOUNT_ID": node_features["ACCOUNT_ID"],
        "account_is_fraud": node_features["account_is_fraud"],
        "appears_in_alert": node_features["ACCOUNT_ID"].isin(alert_involved_accounts),
    }
)

print("\nAccount label vs alert involvement:")
print(
    pd.crosstab(
        alert_coverage["account_is_fraud"],
        alert_coverage["appears_in_alert"],
    ).to_string()
)

alert_coverage.to_csv(OUT / "account_alert_coverage.csv", index=False)

# ---------------------------------------------------------
# 6. Strongly connected component comparison
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("STRONGLY CONNECTED COMPONENT ANALYSIS")
print("=" * 70)

largest_component_id = node_features.groupby("strong_component_id").size().idxmax()

node_features["in_largest_strong_component"] = (
    node_features["strong_component_id"] == largest_component_id
)

print(
    pd.crosstab(
        node_features["in_largest_strong_component"],
        node_features["account_is_fraud"],
        normalize="index",
    )
    .mul(100)
    .round(2)
    .to_string()
)

node_features.to_csv(OUT / "network_fraud_account_features.csv", index=False)

print(f"\nOutputs saved to: {OUT}")
print("Network fraud and alert validation complete.")
