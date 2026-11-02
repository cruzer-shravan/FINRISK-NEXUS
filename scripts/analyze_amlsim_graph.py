from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

ROOT = Path("data/raw/ibm_amlsim/1Kvertices-100Kedges")
OUT = Path("reports/ibm_amlsim")
FIGURES = OUT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

tx = pd.read_csv(ROOT / "transactions.csv")
accounts = pd.read_csv(ROOT / "accounts.csv")
alerts = pd.read_csv(ROOT / "alerts.csv")

# Build a directed, weighted graph.
# Each node is an account; each edge represents one or more transfers.
G = nx.DiGraph()

G.add_nodes_from(accounts["ACCOUNT_ID"])

edges = (
    tx.groupby(["SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID"])
    .agg(
        transaction_count=("TX_ID", "size"),
        total_amount=("TX_AMOUNT", "sum"),
        fraud_transaction_count=("IS_FRAUD", "sum"),
    )
    .reset_index()
)

for row in edges.itertuples(index=False):
    G.add_edge(
        row.SENDER_ACCOUNT_ID,
        row.RECEIVER_ACCOUNT_ID,
        transaction_count=int(row.transaction_count),
        total_amount=float(row.total_amount),
        fraud_transaction_count=int(row.fraud_transaction_count),
    )

print("=" * 65)
print("GRAPH OVERVIEW")
print("=" * 65)
print(f"Nodes (accounts): {G.number_of_nodes():,}")
print(f"Directed edges (unique account pairs): {G.number_of_edges():,}")
print(f"Self-loops: {nx.number_of_selfloops(G):,}")
print(f"Density: {nx.density(G):.6f}")

# Degree: distinct counterparties, not transaction count.
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

metrics = pd.DataFrame(
    {
        "ACCOUNT_ID": list(G.nodes()),
        "in_degree": [in_degree[n] for n in G.nodes()],
        "out_degree": [out_degree[n] for n in G.nodes()],
    }
)

metrics["total_degree"] = metrics["in_degree"] + metrics["out_degree"]

# Account labels are retained for retrospective analysis only.
labels = accounts.set_index("ACCOUNT_ID")["IS_FRAUD"]
metrics["account_is_fraud"] = metrics["ACCOUNT_ID"].map(labels).astype(bool)

print("\n" + "=" * 65)
print("DEGREE ANALYSIS")
print("=" * 65)

print("\nIn-degree summary:")
print(metrics["in_degree"].describe().to_string())

print("\nOut-degree summary:")
print(metrics["out_degree"].describe().to_string())

print("\nTop 10 accounts by in-degree:")
print(metrics.sort_values("in_degree", ascending=False).head(10).to_string(index=False))

print("\nTop 10 accounts by out-degree:")
print(metrics.sort_values("out_degree", ascending=False).head(10).to_string(index=False))

# Connected components.
weak_components = list(nx.weakly_connected_components(G))
strong_components = list(nx.strongly_connected_components(G))

print("\n" + "=" * 65)
print("CONNECTIVITY ANALYSIS")
print("=" * 65)
print(f"Weakly connected components: {len(weak_components):,}")
print(f"Largest weak component: {max(map(len, weak_components)):,}")
print(f"Strongly connected components: {len(strong_components):,}")
print(f"Largest strong component: {max(map(len, strong_components)):,}")

# Betweenness centrality is calculated on the unweighted graph.
# This measures shortest-path brokerage, not financial risk.
centrality = nx.betweenness_centrality(G, normalized=True)

metrics["betweenness_centrality"] = metrics["ACCOUNT_ID"].map(centrality)

print("\n" + "=" * 65)
print("BETWEENNESS CENTRALITY")
print("=" * 65)
print(
    metrics.sort_values("betweenness_centrality", ascending=False).head(10).to_string(index=False)
)

# Calculate transaction involvement separately from account labels.
fraud_tx = tx[tx["IS_FRAUD"].astype(bool)]

fraud_involvement = set(fraud_tx["SENDER_ACCOUNT_ID"]) | set(fraud_tx["RECEIVER_ACCOUNT_ID"])

metrics["fraud_tx_involvement"] = metrics["ACCOUNT_ID"].isin(fraud_involvement)

print("\n" + "=" * 65)
print("LABEL VS NETWORK INVOLVEMENT")
print("=" * 65)
print(
    pd.crosstab(
        metrics["account_is_fraud"],
        metrics["fraud_tx_involvement"],
        rownames=["Account labelled fraud"],
        colnames=["Involved in fraud-labelled transaction"],
    ).to_string()
)

# Persist node-level metrics.
metrics.to_csv(OUT / "graph_account_metrics.csv", index=False)

# Persist component sizes.
component_summary = pd.DataFrame(
    {
        "component_type": (["weak"] * len(weak_components) + ["strong"] * len(strong_components)),
        "component_size": ([len(c) for c in weak_components] + [len(c) for c in strong_components]),
    }
)
component_summary.to_csv(OUT / "graph_component_sizes.csv", index=False)

# Plot degree distributions.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(metrics["in_degree"], bins=30)
axes[0].set_title("In-degree distribution")
axes[0].set_xlabel("Distinct incoming counterparties")
axes[0].set_ylabel("Accounts")

axes[1].hist(metrics["out_degree"], bins=30)
axes[1].set_title("Out-degree distribution")
axes[1].set_xlabel("Distinct outgoing counterparties")
axes[1].set_ylabel("Accounts")

fig.tight_layout()
fig.savefig(FIGURES / "graph_degree_distributions.png", dpi=150)
plt.close(fig)

# Alert type counts for comparison with graph findings.
print("\nAlert types:")
print(alerts["ALERT_TYPE"].value_counts().to_string())

print(f"\nSaved graph metrics to: {OUT}")
print("Graph analysis complete.")
