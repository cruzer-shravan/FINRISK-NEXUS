# PaySim Dataset — Acquisition & Exploration Report

**Project:** FINRISK NEXUS
**Dataset:** PaySim Mobile Money Transactions
**Status:** Initial acquisition and exploratory analysis completed
**Data policy:** Original dataset preserved unchanged

---

## 1. Purpose

This document records the acquisition, schema verification, initial data-quality audit, fraud-label analysis, transaction amount and balance exploration, and temporal analysis of the PaySim dataset.

The objective is to understand the dataset and its limitations before proceeding to feature engineering, model training, and deployment validation.

## 2. Dataset Acquisition

### 2.1 Local file

- **Directory:** `data/raw/paysim/`
- **File:** `PS_20174392719_1491204439457_log.csv`
- **File size:** 470.67 MB
- **Format:** CSV
- **Acquisition status:** File located and readable

The original dataset is retained in the raw-data directory. Exploration scripts read the original file without intentionally modifying it.

### 2.2 Data handling principles

- Preserve original source data.
- Store generated reports separately from raw data.
- Perform large-file analysis in chunks to control memory consumption.
- Document assumptions and limitations before feature engineering.
- Keep raw datasets excluded from Git version control.

## 3. Schema and Structure

### 3.1 Verified dimensions

| Property | Result |
|---|---:|
| Total transactions | 6,362,620 |
| Number of columns | 11 |
| Distinct simulation steps | 743 |
| Missing values detected | 0 |
| Expected column names matched | Yes |

### 3.2 Column definitions

| Column | Observed data type | Description |
|---|---|---|
| `step` | `int64` | Simulation time index |
| `type` | `object` | Transaction category |
| `amount` | `float64` | Transaction amount |
| `nameOrig` | `object` | Sender identifier |
| `oldbalanceOrg` | `float64` | Sender balance before transaction |
| `newbalanceOrig` | `float64` | Sender balance after transaction |
| `nameDest` | `object` | Recipient identifier |
| `oldbalanceDest` | `float64` | Recipient balance before transaction |
| `newbalanceDest` | `float64` | Recipient balance after transaction |
| `isFraud` | `int64` | Synthetic fraud label |
| `isFlaggedFraud` | `int64` | Existing rule-based flag |

The identifiers are stored as strings because they represent entities, not numerical measurements.

### 3.3 Target and detection signal

- `isFraud`: intended supervised-learning target.
- `isFlaggedFraud`: existing rule-based detection signal.

These columns have different purposes and must not be treated as interchangeable.

## 4. Initial Data-Quality Audit

### 4.1 Results

| Check | Result | Interpretation |
|---|---:|---|
| Expected columns | 11 of 11 | Passed |
| Missing values | 0 | None detected |
| Negative transaction amounts | 0 | None detected |
| Invalid binary labels | 0 | None detected |
| Duplicate rows within individual chunks | 0 | No within-chunk duplicates detected |
| Transaction types | 5 | Categories identified |
| Minimum amount | 0.00 | Requires interpretation |
| Maximum amount | 92,445,516.64 | Extreme value worth investigating |
| Minimum step | 1 | Observed |
| Maximum step | 743 | Observed |

Transaction categories:

- `CASH_IN`
- `CASH_OUT`
- `DEBIT`
- `PAYMENT`
- `TRANSFER`

**Limitations:** The duplicate check does not detect duplicates across chunk boundaries. Zero missing values and matching column names do not establish that all records are semantically valid.

## 5. Fraud Class Distribution

### 5.1 Overall distribution

| Label | Transactions | Percentage |
|---|---:|---:|
| Legitimate-labelled (`isFraud = 0`) | 6,354,407 | 99.870918% |
| Fraud-labelled (`isFraud = 1`) | 8,213 | 0.129082% |
| **Total** | **6,362,620** | **100%** |

The dataset is severely imbalanced. There are approximately 775 transactions per fraud-labelled transaction.

A classifier predicting every transaction as legitimate would achieve approximately 99.87% accuracy while detecting no fraud.

Therefore, accuracy alone is not a suitable evaluation metric. Future model evaluation should include precision, recall, F1 score, precision-recall AUC, ROC-AUC, calibration, and an appropriate decision threshold.

## 6. Existing Rule-Based Fraud Flag Evaluation

### 6.1 Confusion matrix

The existing flag was evaluated against `isFraud`.

| Outcome | Count |
|---|---:|
| True negatives | 6,354,407 |
| False positives | 0 |
| False negatives | 8,197 |
| True positives | 16 |
| Total | 6,362,620 |

### 6.2 Performance

| Metric | Result |
|---|---:|
| Precision | 100% |
| Recall | 0.1948% |
| Flagged transactions | 16 |
| Fraud-labelled transactions missed | 8,197 |

The rule flagged only 16 of 8,213 fraud-labelled transactions.

Its observed precision is 100% on this dataset, but its recall is extremely low. The small number of flagged transactions means this precision should not be assumed to generalize to future data.

The rule provides a useful comparison baseline, but it is not sufficient as a standalone fraud detector.

## 7. Fraud by Transaction Type

| Transaction type | Total transactions | Fraud-labelled | Fraud rate |
|---|---:|---:|---:|
| CASH_IN | 1,399,284 | 0 | 0.0000% |
| CASH_OUT | 2,237,500 | 4,116 | 0.1840% |
| DEBIT | 41,432 | 0 | 0.0000% |
| PAYMENT | 2,151,495 | 0 | 0.0000% |
| TRANSFER | 532,909 | 4,097 | 0.7688% |

### Findings

- TRANSFER has the highest observed fraud rate.
- CASH_OUT has the highest absolute number of fraud-labelled transactions.
- The two categories together account for all 8,213 fraud-labelled transactions.
- No fraud-labelled transactions were observed in CASH_IN, DEBIT, or PAYMENT.

These findings describe the synthetic dataset. They do not establish that the other transaction types are inherently fraud-free in real financial systems.

## 8. Transaction Amount Exploration

### 8.1 Amount statistics by fraud label

| Statistic | Legitimate-labelled | Fraud-labelled |
|---|---:|---:|
| Transaction count | 6,354,407 | 8,213 |
| Mean amount | 178,197.04 | 1,467,967.30 |
| Minimum amount | 0.01 | 0.00 |
| Maximum amount | 92,445,516.64 | 10,000,000.00 |
| Zero-amount transactions | 0 | 16 |

The mean transaction amount for fraud-labelled records is approximately 8.24 times the mean for legitimate-labelled records.

This is a descriptive association, not proof that transaction amount alone is sufficient to detect fraud. Extreme values may influence the mean, so medians and percentiles should be examined before final feature decisions.

### 8.2 Mean amount by transaction type and label

| Transaction type | Label | Count | Mean amount |
|---|---:|---:|---:|
| CASH_IN | 0 | 1,399,284 | 168,920.24 |
| CASH_OUT | 0 | 2,233,384 | 173,917.16 |
| CASH_OUT | 1 | 4,116 | 1,455,102.59 |
| DEBIT | 0 | 41,432 | 5,483.67 |
| PAYMENT | 0 | 2,151,495 | 13,057.60 |
| TRANSFER | 0 | 528,812 | 906,229.01 |
| TRANSFER | 1 | 4,097 | 1,480,891.67 |

Fraud-labelled transactions have higher mean amounts within both CASH_OUT and TRANSFER.

Further work is required to understand distributions, overlap between classes, and the influence of extreme transactions.

## 9. Zero-Amount Transaction Investigation

Exactly 16 zero-amount transactions were identified.

All 16 share these characteristics:

- Transaction type: `CASH_OUT`
- `isFraud = 1`
- `isFlaggedFraud = 0`
- Sender balance before transaction: 0
- Sender balance after transaction: 0
- Sender balance change: 0
- Recipient balance change: 0

The records are unusual but cannot automatically be classified as corrupted data. Their meaning should be interpreted in the context of PaySim's synthetic transaction-generation process.

**Modelling implication:** Zero amount may be an investigative signal, but it must not become an automatic fraud rule without further validation.

## 10. Balance Consistency Diagnostics

The initial exploration tested two simple relationships:

1. Sender balance before transaction minus amount equals sender balance after transaction.
2. Recipient balance before transaction plus amount equals recipient balance after transaction.

The observed mismatch counts were:

| Type | Transactions | Sender mismatches | Recipient mismatches |
|---|---:|---:|---:|
| CASH_IN | 1,399,284 | 1,399,284 | 1,399,284 |
| CASH_OUT | 2,237,500 | 1,991,269 | 507,426 |
| DEBIT | 41,432 | 12,462 | 9,631 |
| PAYMENT | 2,151,495 | 1,165,894 | 2,151,495 |
| TRANSFER | 532,909 | 508,782 | 120,811 |

These are preliminary diagnostic counts using an absolute residual tolerance of 0.01.

The equations are not universal validity rules for every PaySim transaction type. The mismatch counts must not be interpreted as counts of invalid records or fraudulent transactions.

Transaction-type-specific balance semantics need further investigation before residuals can be used for cleaning, feature engineering, or fraud classification.

## 11. Temporal Exploration

### 11.1 Overall results

| Metric | Result |
|---|---:|
| Observed steps | 743 |
| Step range | 1–743 |
| Overall fraud rate | 0.129082% |
| Median transactions per step | 529 |
| Median fraud count per step | 10 |
| Steps with zero fraud | 2 |
| Steps with at least one fraud | 741 |

### 11.2 Important temporal checkpoints

| Step | Transactions | Fraud-labelled | Fraud rate |
|---|---:|---:|---:|
| 19 | 51,352 | 11 | 0.021421% |
| 28 | 4 | 4 | 100.0000% |
| 212 | 34,047 | 40 | 0.117485% |
| 743 | 8 | 8 | 100.0000% |

Step 19 has the highest transaction volume. Step 212 has the highest fraud count. Step 28 has a 100% fraud rate, but it contains only four transactions.

The highest fraud rates are concentrated in several low-volume steps. These rates must be interpreted alongside transaction counts rather than treated as reliable evidence of broad fraud hotspots.

### 11.3 Implications for model validation

The temporal results show substantial variation in transaction volume and fraud prevalence.

A stratified random split and a chronological split should be compared before selecting the final evaluation design. A chronological split is particularly relevant if the model's intended use is predicting fraud on future transactions.

Each partition must be checked for sufficient fraud examples. The final split should reflect the deployment scenario rather than being chosen solely on the basis of observed temporal patterns.

## 12. Current Project Artifacts

### Scripts

| Script | Purpose |
|---|---|
| `scripts/audit_paysim.py` | Full-file schema and data-quality audit |
| `scripts/audit_paysim_labels.py` | Fraud labels, existing rule flags, and fraud by transaction type |
| `scripts/explore_paysim_amounts.py` | Amount and balance diagnostics |
| `scripts/investigate_paysim_zero_amounts.py` | Inspection of zero-amount transactions |
| `scripts/explore_paysim_temporal.py` | Temporal aggregation, metrics, and plots |

### Generated reports

| Artifact | Purpose |
|---|---|
| `reports/data_exploration/paysim_data_quality_report.json` | Initial data-quality results |
| `reports/data_exploration/paysim_label_audit.json` | Label-versus-flag evaluation |
| `reports/data_exploration/paysim_amount_balance_report.json` | Amount and balance analysis |
| `reports/data_exploration/paysim_zero_amount_transactions.csv` | Zero-amount records |
| `reports/data_exploration/paysim_temporal_summary.csv` | Per-step temporal metrics |
| `reports/data_exploration/paysim_temporal_report.json` | Temporal summary statistics |
| `reports/data_exploration/figures/paysim_temporal_volume_fraud.png` | Transaction volume and fraud counts |
| `reports/data_exploration/figures/paysim_temporal_fraud_rate.png` | Fraud rate and rolling trend |

## 13. Outstanding Work

The following tasks remain open:

1. Complete duplicate detection across chunk boundaries.
2. Examine amount medians, percentiles, and class overlap.
3. Investigate balance behaviour using transaction-type-specific semantics.
4. Examine sender and recipient identifiers and their potential network structure.
5. Assess temporal patterns and the suitability of chronological validation.
6. Review potential target leakage and determine which features are available at scoring time.
7. Establish a baseline model and appropriate evaluation metrics.
8. Define a defensible feature set and validation strategy before model tuning.

## 14. Conclusion

The initial PaySim exploration confirms that the dataset contains 6.36 million transactions, 11 expected columns, 8,213 fraud-labelled transactions, and only 16 transactions flagged by the existing rule.

Fraud-labelled transactions are concentrated in CASH_OUT and TRANSFER categories, have a substantially higher mean amount, and exhibit temporal patterns that warrant careful validation.

The data is suitable for continued investigation, but the current findings are exploratory. Balance diagnostics, temporal patterns, and synthetic-data characteristics must be understood before they are used to make modelling or production decisions.

**Next phase:** Complete the remaining data investigations, define the feature availability boundary, and establish a defensible fraud-detection baseline before model optimization.
