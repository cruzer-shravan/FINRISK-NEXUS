# FINRISK NEXUS — PaySim Gold Layer

**Project:** FINRISK NEXUS
**Dataset:** PaySim synthetic financial transactions
**Layer:** Gold
**Status:** Initial pipeline execution completed; recorded validation checks passed
**Source layer:** Silver
**Implementation:** Polars lazy processing and Parquet
**Last updated:** 19 October 2026

---

## 1. Objective

The PaySim Gold layer transforms the validated Silver transaction dataset into analytical datasets suitable for fraud analysis, feature engineering, reporting, and subsequent machine-learning experimentation.

The Gold layer is implemented separately from Silver to preserve the validated input data and provide a reproducible downstream transformation.

The initial implementation produces:

1. Transaction-level features.
2. Aggregated fraud analytics.
3. A machine-readable data quality and validation report.

## 2. Architecture and data flow

```text
PaySim Source Dataset
        |
        v
Bronze Layer
  Raw Parquet
        |
        v
Silver Layer
  Validated Transactions
  6,362,620 rows
        |
        v
Gold Pipeline
  Polars Lazy Processing
        |
        +---------------------------+
        |                           |
        v                           v
Transaction Features          Fraud Analytics
        |                           |
        +-------------+-------------+
                      |
                      v
             Validation Report
```

The Silver dataset remains the input source and is not modified by the Gold pipeline.

## 3. Input dataset

**Input path**

`data/silver/paysim/transactions.parquet`

**Verified baseline**

| Attribute | Value |
|---|---:|
| Total transactions | 6,362,620 |
| Source columns | 11 |
| Legitimate transactions | 6,354,407 |
| Fraud-labelled transactions | 8,213 |
| Fraud prevalence | Approximately 0.1291% |
| Negative transaction amounts | 0 |
| Rows with negative balances | 0 |

### 3.1 Source schema

| Column | Data type | Purpose |
|---|---|---|
| `step` | BIGINT | Simulation time step |
| `type` | VARCHAR | Transaction type |
| `amount` | DOUBLE | Transaction amount |
| `nameOrig` | VARCHAR | Origin account identifier |
| `oldbalanceOrg` | DOUBLE | Origin balance before transaction |
| `newbalanceOrig` | DOUBLE | Origin balance after transaction |
| `nameDest` | VARCHAR | Destination account identifier |
| `oldbalanceDest` | DOUBLE | Destination balance before transaction |
| `newbalanceDest` | DOUBLE | Destination balance after transaction |
| `isFraud` | TINYINT | Fraud target label |
| `isFlaggedFraud` | TINYINT | Existing simulator flag |

The `step` column represents simulation time steps. It is not a calendar timestamp.

## 4. Gold output datasets

### 4.1 Transaction features

**Output path**

`data/gold/paysim/transaction_features.parquet`

The dataset preserves all 6,362,620 source transactions and retains the original 11 columns. Five derived features are added, producing 16 columns.

| Derived feature | Definition | Purpose |
|---|---|---|
| `origin_balance_error` | `oldbalanceOrg - newbalanceOrig - amount` | Measures the difference between the origin balance reduction and transaction amount |
| `destination_balance_error` | `newbalanceDest - oldbalanceDest - amount` | Measures the difference between the destination balance increase and transaction amount |
| `origin_balance_mismatch` | 1 when the absolute origin error exceeds 0.01; otherwise 0 | Indicates an origin balance discrepancy |
| `destination_balance_mismatch` | 1 when the absolute destination error exceeds 0.01; otherwise 0 | Indicates a destination balance discrepancy |
| `log_amount` | `ln(1 + amount)` | Provides a logarithmic transformation of transaction amount |

The mismatch tolerance is `0.01`.

Balance discrepancies are retained as analytical signals rather than automatically classified as invalid data or fraud. Their predictive value must be established through subsequent analysis.

### 4.2 Fraud analytics

**Output path**

`data/gold/paysim/fraud_analytics.parquet`

The analytics dataset contains aggregated transaction counts grouped by:

- Transaction type.
- Simulation time step.

The recorded aggregation fields are:

| Field | Description |
|---|---|
| `summary_level` | Identifies the aggregation level |
| `group_value` | Transaction type or simulation step |
| `total_transactions` | Number of transactions in the group |
| `fraud_transactions` | Number of fraud-labelled transactions |
| `flagged_transactions` | Number of transactions carrying the simulator flag |

The transaction-type aggregation is validated against the total Silver row count.

## 5. Fraud distribution findings

The Silver exploration established the following distribution:

| Transaction type | Total transactions | Fraud transactions | Fraud rate |
|---|---:|---:|---:|
| CASH_OUT | 2,237,500 | 4,116 | 0.183955% |
| PAYMENT | 2,151,495 | 0 | 0% |
| CASH_IN | 1,399,284 | 0 | 0% |
| TRANSFER | 532,909 | 4,097 | 0.768799% |
| DEBIT | 41,432 | 0 | 0% |
| **Total** | **6,362,620** | **8,213** | **Approximately 0.1291%** |

All fraud-labelled transactions in this dataset occur in the CASH_OUT and TRANSFER categories. This is an observed property of the PaySim dataset, not a universal rule about financial fraud.

The dataset is highly imbalanced. Subsequent model evaluation should consider precision, recall, PR-AUC, calibration, and the business costs of false positives and false negatives rather than relying on accuracy alone.

## 6. Existing fraud flag analysis

The Silver exploration found:

| `isFraud` | `isFlaggedFraud` | Transactions |
|---:|---:|---:|
| 0 | 0 | 6,354,407 |
| 1 | 0 | 8,197 |
| 1 | 1 | 16 |

Only 16 of the 8,213 fraud-labelled transactions were flagged by the existing simulator flag.

The Gold dataset retains `isFlaggedFraud` for auditing. It is excluded from the declared initial model feature list so that initial modelling does not depend on this existing flag.

## 7. Model feature and target separation

The initial declared model feature list contains 12 columns:

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- `origin_balance_error`
- `destination_balance_error`
- `origin_balance_mismatch`
- `destination_balance_mismatch`
- `log_amount`

**Target:** `isFraud`

**Audit-only field:** `isFlaggedFraud`

The account identifiers `nameOrig` and `nameDest` remain available in the stored Gold transaction dataset but are not part of the initial declared feature list.

The Gold Parquet file is an analytical dataset, not a ready-to-fit model matrix. Any future training pipeline must explicitly select the approved features and target, perform appropriate preprocessing, and use a leakage-safe evaluation strategy.

In particular, historical account-behaviour features must be constructed using only information available at the prediction time.

## 8. Data quality and validation

**Validation report**

`reports/data_quality/paysim_gold_validation.json`

The report records the source and output paths, row counts, column counts, fraud-label counts, source invariant checks, feature definitions, SHA-256 checksums, and pipeline execution timestamp.

### 8.1 Recorded validation results

| Check | Expected result | Recorded result |
|---|---|---|
| Transaction row count preservation | Gold row count equals Silver row count | Passed |
| Feature schema | Expected 16-column schema | Passed |
| Fraud-label count preservation | Label counts unchanged | Passed |
| Transaction-type analytics reconciliation | Aggregate transaction count equals Silver row count | Passed |
| Null transaction amounts | 0 | Passed |
| Negative transaction amounts | 0 | Passed |
| Null fraud labels | 0 | Passed |
| Invalid fraud labels | 0 |
| Null simulator flags | 0 |
| Invalid simulator flags | 0 |

**Recorded pipeline status:** `passed`

The balance mismatch analysis is descriptive and does not impose a requirement that all balances reconcile exactly.

### 8.2 Reproducibility

SHA-256 checksums are recorded for:

- The Silver source Parquet file.
- The Gold transaction features Parquet file.
- The Gold fraud analytics Parquet file.

These checksums support integrity verification and comparison of artifacts across runs.

The pipeline refuses to overwrite existing output artifacts. Review or archive existing outputs before a deliberate rerun.

## 9. Implementation files

| Component | Path |
|---|---|
| Silver pipeline | `src/pipelines/paysim_silver_pipeline.py` |
| Gold pipeline | `src/pipelines/paysim_gold_pipeline.py` |
| Gold runner | `scripts/build_paysim_gold.py` |
| Transaction features | `data/gold/paysim/transaction_features.parquet` |
| Fraud analytics | `data/gold/paysim/fraud_analytics.parquet` |
| Validation report | `reports/data_quality/paysim_gold_validation.json` |
| Silver validation record | `docs/data_validation/FINRISK_NEXUS_PaySim_Silver_Validation_Freeze.md` |

## 10. Execution instructions

Run from the FINRISK NEXUS project root with the project virtual environment activated.

```powershell
python scripts/build_paysim_gold.py
```

Inspect the validation report:

```powershell
Get-Content ".\reports\data_quality\paysim_gold_validation.json"
```

The pipeline should be treated as successfully validated only when the output report confirms that all required checks have passed.

## 11. Limitations and next steps

The current Gold implementation establishes a reproducible transaction-level feature dataset and basic fraud aggregations. It does not yet establish model performance or production readiness.

Recommended next steps:

1. Independently verify the Gold schemas, row counts, and fraud-label distribution.
2. Verify feature formulas and representative records against Silver.
3. Validate the aggregated fraud analytics at both transaction-type and simulation-step levels.
4. Add explicit automated tests for the Gold pipeline.
5. Design leakage-safe historical account and transaction behaviour features.
6. Create a reproducible train, validation, and test strategy.
7. Establish baseline fraud models and evaluate them with imbalance-aware metrics.
8. Document and freeze the Gold artifacts after the required checks and review are complete.

## 12. Current status

The initial Gold pipeline execution completed successfully, producing the two Parquet outputs and the JSON validation report.

The report records a `passed` validation status, with row preservation, schema matching, fraud-label preservation, and transaction-type reconciliation all passing.

This status applies to the checks implemented in the initial pipeline. Independent verification, expanded automated testing, and model evaluation remain separate activities.

**End of document**
