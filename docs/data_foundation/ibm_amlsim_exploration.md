# IBM AMLSim — Dataset, Transaction and Network Exploration

**Project:** FINRISK NEXUS
**Dataset:** IBM AMLSim — `1Kvertices-100Kedges`
**Status:** Exploratory analysis completed; modelling not yet started

## 1. Objective

Explore the IBM AMLSim synthetic financial transaction dataset to understand its structure, data quality, transaction behaviour, account relationships, fraud labels and alert coverage.

The findings will guide feature engineering and help determine whether network information adds measurable value to financial-crime risk prediction.

## 2. Dataset Overview

The dataset contains three primary files:

| File | Records | Purpose |
|---|---:|---|
| `accounts.csv` | 1,000 | Account attributes and account-level fraud labels |
| `transactions.csv` | 117,533 | Transaction activity and transaction-level fraud labels |
| `alerts.csv` | 175 | Fraud-related alerts and associated transaction information |

Additional configuration files describe account characteristics, alert patterns, transaction types and network statistics.

## 3. Data Quality and Relationship Validation

The initial validation established the following:

- No duplicate account IDs were detected.
- No duplicate transaction IDs were detected.
- No missing values were reported in the inspected primary tables.
- All transaction sender and receiver account references matched known accounts.
- All alert transaction references matched existing transactions.
- All 175 alert records matched transaction records, with no fraud-label disagreements.
- The transaction-level dataset contained 117,358 non-fraud-labelled transactions and 175 fraud-labelled transactions.

These checks support the structural consistency of the inspected data. They do not, by themselves, establish that the dataset is representative of real financial activity.

## 4. Transaction Behaviour

Key observations:

- Total transactions: 117,533.
- Unique sending accounts: 1,000.
- Unique receiving accounts: 986.
- Fraud-labelled transactions: 175.
- Observed transaction fraud rate: approximately 0.149%.
- Zero-amount transactions: 3.
- Negative transaction amounts: 0.
- Negative timestamps: 0.
- Observed timestamp range: 0–199.
- All transactions in the inspected sample were classified as `TRANSFER`.

The transaction amounts are highly dispersed, with a reported maximum of approximately 11.48 million. This warrants further investigation during feature engineering and model evaluation.

**Implication:** Transaction amount, transaction frequency, account activity and temporal behaviour are candidate features. The timestamp represents a simulation index and should not automatically be interpreted as a real calendar timestamp.

## 5. Account Network Exploration

The directed account graph was constructed from transaction relationships.

| Network metric | Observed result |
|---|---:|
| Account nodes | 1,000 |
| Unique directed account pairs | 5,891 |
| Self-loops | 8 |
| Network density | 0.005897 |
| Weakly connected components | 1 |
| Largest weakly connected component | 1,000 accounts |
| Strongly connected components | 26 |

The network analysis examined in-degree, out-degree, total degree, betweenness centrality, transaction counts, transaction amounts and connected-component membership.

Accounts with high network degree or centrality may be useful candidates for additional investigation. However, high connectivity is not itself evidence of fraud.

## 6. Fraud and Network Findings

The account-level dataset contains:

- 162 fraud-labelled accounts.
- 838 non-fraud-labelled accounts.
- 159 fraud-labelled accounts appearing in alerts.
- 3 fraud-labelled accounts absent from alerts.

The network-ranking validation reported the following fraud prevalence among the highest-ranked accounts:

| Ranking feature | Top 1% | Top 5% | Top 10% |
|---|---:|---:|---:|
| In-degree | 90% | 52% | 46% |
| Out-degree | 60% | 50% | 37% |
| Total degree | 80% | 60% | 47% |
| Betweenness centrality | 80% | 64% | 45% |
| Total transaction count | 60% | 56% | 43% |
| Total amount moved | 10% | 20% | 16% |

These figures are exploratory results from the current synthetic dataset. They are not estimates of real-world fraud detection performance.

The observed patterns justify testing network features in a controlled modelling experiment. The feature-ranking methodology, baseline fraud prevalence, tie handling and possible effects of synthetic data generation must be considered before drawing stronger conclusions.

## 7. Alert Analysis

The alert dataset contains 175 records across 40 unique alert IDs.

| Alert type | Records |
|---|---:|
| `cycle` | 96 |
| `fan_in` | 79 |

The observed alert records are associated with fraud-labelled transactions. Alert IDs can appear in multiple rows because a single alert may cover several transactions.

Alert coverage is therefore not equivalent to independent fraud detection performance. The dataset's alert-generation mechanism and the relationship between alerts and fraud labels require consideration before alerts are used as model inputs or evaluation targets.

## 8. Network Validation and Fraud Association

### 8.1 Network Graph Analysis

The account network was analysed using NetworkX to examine graph connectivity, node degrees, betweenness centrality and the relationship between account connectivity and fraud labels.

Key findings:

- The graph contains 1,000 account nodes and 5,891 unique directed account pairs.
- The graph has one weakly connected component containing all 1,000 accounts.
- There are 26 strongly connected components.
- Accounts with high in-degree, out-degree, total degree and betweenness centrality were examined for fraud-label concentration.

### 8.2 Fraud Prevalence Among Highly Ranked Accounts

The network-ranking analysis showed elevated fraud prevalence among several groups of highly ranked accounts.

For example, the top 1% ranked by in-degree contained 9 fraud-labelled accounts among 10 selected accounts, while the top 1% ranked by total degree contained 8.

These findings suggest that network connectivity may provide useful signals for prioritising accounts for investigation. However, these rankings are exploratory and do not establish predictive performance.

### 8.3 Fraud and Alert Association

The validation established the following:

- Total fraud-labelled accounts: 162.
- Fraud-labelled accounts appearing in alerts: 159.
- Fraud-labelled accounts absent from alerts: 3.
- Total alert records: 175.
- Fraud-label disagreements between matched alert and transaction records: 0.

The alert data contains two observed alert types: `cycle` and `fan_in`.

Alert coverage should not be interpreted as independent fraud detection performance because the relationship between alert generation and the dataset's fraud labels may be influenced by the simulation process.

### 8.4 Feature Engineering Implications

The findings justify testing the following network-derived features:

1. Account in-degree and out-degree.
2. Total degree and betweenness centrality.
3. Incoming and outgoing transaction counts and amounts.
4. Number of unique counterparties.
5. Transaction activity over defined time windows.
6. Network component membership.

All features must be calculated using information available at the intended prediction time. Fraud labels, future transactions and alert-derived information must be excluded where they would cause target leakage.

### 8.5 Decision

The network exploration provides sufficient justification to proceed to baseline modelling and a controlled network-feature experiment. It does not justify assuming that network features will improve model performance.

The next milestone is to compare a transaction/account behavioural baseline against a network-enhanced model using the same leakage-safe evaluation design.

## 9. Implications for Feature Engineering

The exploration identified the following candidate feature groups:

1. **Transaction features:** amount, transaction frequency and transaction activity.
2. **Account behaviour:** incoming and outgoing transaction counts and amounts.
3. **Counterparty behaviour:** number and diversity of counterparties.
4. **Network features:** in-degree, out-degree, total degree and betweenness centrality.
5. **Temporal features:** transaction activity across defined time windows.
6. **Network structure:** connected-component membership and potentially suspicious transaction patterns.

These are candidate features, not confirmed predictors. Their value must be established through out-of-sample evaluation.

## 10. Modelling and Validation Plan

The next steps are:

1. Define the initial prediction target as transaction-level fraud prediction.
2. Establish the prediction-time boundary and audit possible label leakage.
3. Create a reproducible baseline using transaction and account behavioural features.
4. Add appropriate network features calculated using information available at prediction time.
5. Compare the baseline and network-enhanced models on the same held-out evaluation data.
6. Evaluate precision, recall, F1, PR-AUC, confusion matrices and operational false-positive burden.
7. Retain network features only when they provide demonstrable improvement.
8. Evaluate temporal generalization and document limitations arising from synthetic data.

Account-level risk scoring and more advanced network-based fraud-ring detection can be developed as subsequent extensions.

## 11. Limitations

- IBM AMLSim is synthetic and does not establish how a model will perform on real banking data.
- Account and transaction fraud labels represent different prediction targets.
- Alert coverage may reflect the simulation's construction rather than independent detection effectiveness.
- Network features must not incorporate future transactions or information derived from target labels.
- High centrality, transaction volume or network connectivity does not establish criminal activity.
- No model performance improvement has yet been demonstrated by this exploration.

## 12. Conclusion

The IBM AMLSim exploration established a structural understanding of the dataset and identified candidate transaction, behavioural and network features for financial-crime modelling.

The most important next step is to move from descriptive analysis to controlled experiments. A transaction-level baseline will be compared with a network-enhanced model to determine whether network intelligence provides measurable predictive value.

**Current milestone:** Dataset exploration and initial network validation completed. Feature engineering and baseline modelling are the next priorities.
