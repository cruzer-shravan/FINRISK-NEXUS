# FINRISK NEXUS — PaySim Temporal Drift Audit

## 1. Audit overview

**Dataset:** `data/gold/paysim/transaction_features.parquet`

**Target:** `isFraud`

**Temporal column:** `step`

**Interval size:** 50 simulation steps

**Total transactions analysed:** 6,362,620

**Analysis type:** Descriptive temporal distribution analysis.

This audit examines transaction volume and observed fraud prevalence
across simulation periods and transaction types.

It does not perform statistical drift testing, model retraining,
threshold optimization, or feature selection.

## 2. Fraud prevalence across modeling periods

| period_group | transactions | fraud_cases | fraud_rate | min_step | max_step |
| --- | --- | --- | --- | --- | --- |
| Early (1-323) | 4463587 | 3643 | 0.081616% | 1 | 323 |
| Validation (324-378) | 980416 | 564 | 0.057527% | 324 | 378 |
| Test (379-743) | 918617 | 4006 | 0.436090% | 379 | 743 |

The test period has a different observed fraud prevalence from the
training and validation periods.

This difference should be considered when interpreting the locked
test evaluation.

## 3. Fraud rate by 50-step interval

| period | transactions | fraud_cases | fraud_rate | unique_steps |
| --- | --- | --- | --- | --- |
| 1 | 1030263 | 590 | 0.057267% | 50 |
| 51 | 29253 | 600 | 2.051072% | 50 |
| 101 | 451852 | 520 | 0.115082% | 50 |
| 151 | 870814 | 544 | 0.062470% | 50 |
| 201 | 866110 | 587 | 0.067774% | 50 |
| 251 | 825358 | 560 | 0.067849% | 50 |
| 301 | 870265 | 500 | 0.057454% | 50 |
| 351 | 843115 | 576 | 0.068318% | 50 |
| 401 | 227687 | 580 | 0.254736% | 50 |
| 451 | 47090 | 504 | 1.070291% | 50 |
| 501 | 104212 | 504 | 0.483630% | 50 |
| 551 | 93028 | 548 | 0.589070% | 50 |
| 601 | 22742 | 578 | 2.541553% | 50 |
| 651 | 69410 | 532 | 0.766460% | 50 |
| 701 | 11421 | 490 | 4.290342% | 43 |

The results show that fraud prevalence varies across simulation
intervals rather than remaining constant throughout the dataset.

The intervals are descriptive groupings and should not be interpreted
as evidence of the cause of the observed changes.

## 4. Fraud rate by transaction type

| type | transactions | fraud_cases | fraud_rate |
| --- | --- | --- | --- |
| TRANSFER | 532909 | 4097 | 0.768799% |
| CASH_OUT | 2237500 | 4116 | 0.183955% |
| CASH_IN | 1399284 | 0 | 0.000000% |
| DEBIT | 41432 | 0 | 0.000000% |
| PAYMENT | 2151495 | 0 | 0.000000% |

The transaction-type analysis shows how fraud cases are distributed
across the available transaction categories.

A zero observed fraud rate does not establish that a transaction
category is inherently incapable of fraud.

## 5. Fraud rate by transaction type and modeling period

| period_group | type | transactions | fraud_cases | fraud_rate |
| --- | --- | --- | --- | --- |
| Early (1-323) | CASH_IN | 977376 | 0 | 0.000000% |
| Early (1-323) | CASH_OUT | 1582647 | 1831 | 0.115692% |
| Early (1-323) | DEBIT | 27764 | 0 | 0.000000% |
| Early (1-323) | PAYMENT | 1506552 | 0 | 0.000000% |
| Early (1-323) | TRANSFER | 369248 | 1812 | 0.490727% |
| Validation (324-378) | CASH_IN | 217839 | 0 | 0.000000% |
| Validation (324-378) | CASH_OUT | 344551 | 282 | 0.081846% |
| Validation (324-378) | DEBIT | 7080 | 0 | 0.000000% |
| Validation (324-378) | PAYMENT | 327506 | 0 | 0.000000% |
| Validation (324-378) | TRANSFER | 83440 | 282 | 0.337967% |
| Test (379-743) | CASH_IN | 204069 | 0 | 0.000000% |
| Test (379-743) | CASH_OUT | 310302 | 2003 | 0.645500% |
| Test (379-743) | DEBIT | 6588 | 0 | 0.000000% |
| Test (379-743) | PAYMENT | 317437 | 0 | 0.000000% |
| Test (379-743) | TRANSFER | 80221 | 2003 | 2.496852% |

This comparison highlights how the observed fraud prevalence for
different transaction types varies across the early, validation,
and test periods.

## 6. Key observations

1. Fraud prevalence varies across the simulation timeline.
2. The validation and test periods have substantially different
   observed fraud rates.
3. Fraud prevalence also differs across transaction types.
4. These observations indicate temporal distribution differences
   that warrant further investigation.
5. The underlying cause of these differences has not been established.

## 7. Methodology and limitations

- Source data was read from the existing PaySim Gold Parquet dataset.
- Only `step`, `type`, and `isFraud` were loaded.
- Temporal intervals were grouped into 50-step periods.
- Modeling-period boundaries were aligned with the existing
  chronological split manifest.
- Fraud prevalence was calculated as fraud cases divided by
  transaction count.
- This is descriptive analysis, not a formal statistical drift test.
- No model predictions or thresholds were changed.
- No training, validation, or test datasets were modified.
- No model was retrained during this audit.

The audit does not establish whether the observed differences are
caused by changes in transaction behavior, simulation design,
fraud-generation mechanisms, or other factors.

## 8. Output artifacts

| Artifact | Description |
|---|---|
| `temporal_drift/fraud_by_50_step_interval.csv` | Fraud statistics by 50-step interval |
| `temporal_drift/fraud_by_transaction_type.csv` | Fraud statistics by transaction type |
| `temporal_drift/fraud_by_period_and_type.csv` | Fraud statistics by period and transaction type |
| `temporal_drift_report.md` | Consolidated temporal drift report |

## 9. Evaluation integrity

This audit is supplementary descriptive analysis.

The existing Decision Tree and Random Forest locked-test evaluation
results remain frozen. The test partition must not be reused for
threshold tuning, feature selection, hyperparameter optimization,
or iterative model selection.

---

**Project:** FINRISK NEXUS

**Dataset:** PaySim
