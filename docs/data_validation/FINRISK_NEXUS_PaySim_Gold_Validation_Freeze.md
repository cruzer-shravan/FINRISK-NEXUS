# FINRISK NEXUS — PaySim Gold Validation Freeze Record

**Project:** FINRISK NEXUS
**Dataset:** PaySim
**Layer:** Gold
**Validation date:** 19 October 2026
**Status:** Independently verified against the checks documented below

## 1. Purpose

This document records the validation of the PaySim Gold data-engineering outputs generated from the frozen Silver transaction dataset.

The Gold pipeline transforms Silver transactions into transaction-level analytical features and aggregated fraud analytics while preserving the original transaction records.

## 2. Source and output artifacts

### Source

`data/silver/paysim/transactions.parquet`

### Gold outputs

- `data/gold/paysim/transaction_features.parquet`
- `data/gold/paysim/fraud_analytics.parquet`

### Validation report

`reports/data_quality/paysim_gold_validation.json`

### Pipeline implementation

- `src/pipelines/paysim_gold_pipeline.py`
- `scripts/build_paysim_gold.py`

## 3. Verified transaction dataset

| Validation item | Verified result |
|---|---|
| Silver source rows | 6,362,620 |
| Gold transaction rows | 6,362,620 |
| Original columns | 11 |
| Gold transaction columns | 16 |
| Legitimate transactions | 6,354,407 |
| Fraud-labelled transactions | 8,213 |
| Fraud-label counts preserved | Yes |

The Gold transaction dataset preserves the full transaction count and the original fraud-label distribution.

## 4. Feature validation

The following derived features were independently checked against their defined formulas.

| Feature | Definition | Result |
|---|---|---|
| `origin_balance_error` | `oldbalanceOrg - newbalanceOrig - amount` | Passed |
| `destination_balance_error` | `newbalanceDest - oldbalanceDest - amount` | Passed |
| `log_amount` | `ln(1 + amount)` | Passed |
| `origin_balance_mismatch` | Absolute origin error greater than 0.01 | Passed |
| `destination_balance_mismatch` | Absolute destination error greater than 0.01 | Passed |

All five feature checks returned zero formula or indicator errors using the independent verification queries.

## 5. Fraud analytics validation

The fraud analytics output was independently inspected.

| Aggregation | Number of groups | Aggregated transaction count |
|---|---:|---:|
| Transaction type | 5 | 6,362,620 |
| Simulation step | 743 | 6,362,620 |

Both aggregation levels reconcile to the full Silver transaction count.

## 6. Recorded pipeline validation

The generated JSON report records the following checks as passed:

- Transaction row count preservation.
- Expected Gold feature schema.
- Fraud-label count preservation.
- Transaction-type analytics reconciliation.

The report also records source invariant checks, artifact paths, and SHA-256 checksums for the source and both Gold Parquet outputs.

The independent checks additionally verified the five derived feature formulas and indicators.

## 7. Fraud distribution baseline

| Label | Transaction count | Approximate proportion |
|---|---:|---:|
| Legitimate | 6,354,407 | 99.8709% |
| Fraud-labelled | 8,213 | 0.1291% |
| **Total** | **6,362,620** | **100%** |

This baseline establishes the class distribution for subsequent fraud-detection experiments.

## 8. Validation scope and limitations

This freeze record covers the current Gold artifacts and the specific checks listed in this document.

It does not establish:

- Predictive performance of a fraud-detection model.
- Production readiness or operational monitoring readiness.
- Absence of every possible data-quality issue.
- General applicability of PaySim fraud patterns to real financial institutions.

The Gold dataset retains `isFraud` as the target and `isFlaggedFraud` as an audit field. The initial declared model feature list excludes both fields from model inputs.

## 9. Reproducibility

The machine-readable validation report is retained at:

`reports/data_quality/paysim_gold_validation.json`

Its SHA-256 checksums should be used to identify the exact source and output artifacts associated with this validation.

Do not overwrite frozen Gold outputs without reviewing the existing artifacts and documenting a new pipeline run and validation record.

## 10. Freeze decision

The initial PaySim Gold outputs have passed the documented row-count, schema, label-preservation, aggregation-reconciliation, and independent feature-formula checks.

**Decision:** Accept the current PaySim Gold artifacts as validated for the scope documented above.

**Next phase:** Build the fraud-modeling dataset and establish leakage-safe training, validation, and test procedures.

---

**End of freeze record**
