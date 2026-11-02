# FINRISK NEXUS — PaySim Fraud Model Comparison

**Report date:** 2 November 2026
**Status:** Consolidated model comparison with supplementary temporal drift audit
**Target:** `isFraud`
**Temporal column:** `step`
**Features:** `amount`, `oldbalanceOrg`, `oldbalanceDest`, `log_amount`, `type`
**Experiment family:** `paysim-tree-fraud-v1` (tree models); `paysim-logreg-baseline-v1` (baseline)

## 1. Executive summary

Random Forest is the strongest of the models with a supplied test evaluation. At its validation-selected threshold of `0.930525`, it achieved **F1 0.8139**, **average precision 0.8939**, and **ROC-AUC 0.9989** on the supplied test partition.

Compared with Decision Tree at its own validation-selected threshold, Random Forest captured **329 additional fraud cases**, generated **75 fewer false positives**, and achieved an F1 improvement of **0.0669**. It flagged 2,992 transactions versus 2,738 for Decision Tree.

A supplementary temporal drift audit identified substantial differences in observed fraud prevalence across chronological modeling periods. Test-period fraud prevalence was approximately **7.58 times** validation-period prevalence.

**Decision:** Prefer Random Forest for the next stage of evaluation, subject to temporal distribution, calibration, stability, and operational validation. This is not a claim of production readiness.

## 2. Evaluation integrity and comparability

- The Logistic Regression artifact supplied here contains validation results only. No Logistic Regression test metrics were provided.
- Decision Tree and Random Forest have explicit test-evaluation artifacts.
- For both tree models, the decision threshold was selected using validation predictions and applied unchanged to the test set.
- The supplied test results have been used to compare Decision Tree and Random Forest. Consequently, the test partition should no longer be described as an untouched holdout.
- The reported test metrics and thresholds are frozen. Do not use this partition for further threshold tuning, feature selection, hyperparameter optimization, or iterative model selection.
- Do not rank Logistic Regression validation metrics against Decision Tree or Random Forest test metrics. Compare models within the same split.
- The temporal drift audit was descriptive. It did not change model predictions, thresholds, or evaluation metrics.

## 3. Validation comparison

| Model | Threshold | Average precision | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.999980 | 0.1700 | 0.9821 | 0.4355 | 0.2216 | 0.2938 |
| Decision Tree | 0.998389 | 0.6239 | 0.9871 | 0.6833 | 0.6082 | 0.6435 |
| Random Forest | 0.930525 | 0.7684 | 0.9991 | 0.8197 | 0.6933 | 0.7512 |

Random Forest has the strongest validation average precision, ROC-AUC, precision, recall, and F1 among the three supplied validation records. This is consistent with—but does not replace—the test comparison below.

## 4. Test evaluation comparison

| Model | Threshold | Average precision | ROC-AUC | Precision | Recall | F1 | TP | FP | TN | FN | Flagged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.998389 | 0.8137 | 0.9878 | 0.9200 | 0.6288 | 0.7470 | 2,519 | 219 | 914,392 | 1,487 | 2,738 |
| Random Forest | 0.930525 | 0.8939 | 0.9989 | 0.9519 | 0.7109 | 0.8139 | 2,848 | 144 | 914,467 | 1,158 | 2,992 |

### Test comparison interpretation

- **Random Forest F1:** 0.8139, versus Decision Tree 0.7470.
- **Fraud capture:** Random Forest recall is 71.09%, versus Decision Tree 62.88%.
- **False positives:** Random Forest produced 144 false positives, versus Decision Tree 219.
- **False negatives:** Random Forest missed 1,158 fraud cases, versus Decision Tree 1,487.
- **Operational trade-off:** Random Forest flagged 2,992 transactions, 254 more than Decision Tree, while capturing more fraud.

These results establish Random Forest as the leading candidate among the supplied test evaluations. They do not establish future-period performance.

## 5. Data splits and temporal distribution analysis

### 5.1 Chronological modeling splits

| Split | Rows | Fraud cases | Fraud prevalence | Simulation steps |
|---|---:|---:|---:|---|
| Train | 4,463,587 | 3,643 | 0.081616% | 1–323 |
| Validation | 980,416 | 564 | 0.057527% | 324–378 |
| Test | 918,617 | 4,006 | 0.436090% | 379–743 |

The test-period fraud prevalence is approximately **7.58 times** the validation-period prevalence.

This difference is important when interpreting the test metrics. A model evaluated on a period with substantially different class prevalence may experience different precision, alert volumes, and operational performance.

### 5.2 Fraud prevalence across 50-step intervals

The temporal audit grouped transactions into intervals of 50 simulation steps.

| Starting step | Transactions | Fraud cases | Fraud rate |
|---:|---:|---:|---:|
| 1 | 1,030,263 | 590 | 0.057267% |
| 51 | 29,253 | 600 | 2.051072% |
| 101 | 451,852 | 520 | 0.115082% |
| 151 | 870,814 | 544 | 0.062470% |
| 201 | 866,110 | 587 | 0.067774% |
| 251 | 825,358 | 560 | 0.067849% |
| 301 | 870,265 | 500 | 0.057454% |
| 351 | 843,115 | 576 | 0.068318% |
| 401 | 227,687 | 580 | 0.254736% |
| 451 | 47,090 | 504 | 1.070291% |
| 501 | 104,212 | 504 | 0.483630% |
| 551 | 93,028 | 548 | 0.589070% |
| 601 | 22,742 | 578 | 2.541553% |
| 651 | 69,410 | 532 | 0.766460% |
| 701 | 11,421 | 490 | 4.290342% |

The observed fraud rate varies substantially across the simulation timeline. The highest reported interval rate is 4.290342% for steps 701–743, which contains 11,421 transactions.

The interval results are descriptive. Smaller transaction volumes in some intervals should be considered when interpreting their fraud rates.

### 5.3 Fraud distribution by transaction type

| Transaction type | Transactions | Fraud cases | Fraud rate |
|---|---:|---:|---:|
| TRANSFER | 532,909 | 4,097 | 0.768799% |
| CASH_OUT | 2,237,500 | 4,116 | 0.183955% |
| CASH_IN | 1,399,284 | 0 | 0.000000% |
| DEBIT | 41,432 | 0 | 0.000000% |
| PAYMENT | 2,151,495 | 0 | 0.000000% |

The observed fraud cases are concentrated in TRANSFER and CASH_OUT transactions.

Zero observed fraud cases in CASH_IN, DEBIT, and PAYMENT do not establish that these transaction categories are inherently incapable of fraud.

### 5.4 Transaction-type fraud prevalence across periods

| Period | Transaction type | Transactions | Fraud cases | Fraud rate |
|---|---|---:|---:|---:|
| Early (1–323) | CASH_OUT | 1,582,647 | 1,831 | 0.115692% |
| Early (1–323) | TRANSFER | 369,248 | 1,812 | 0.490727% |
| Validation (324–378) | CASH_OUT | 344,551 | 282 | 0.081846% |
| Validation (324–378) | TRANSFER | 83,440 | 282 | 0.337967% |
| Test (379–743) | CASH_OUT | 310,302 | 2,003 | 0.645500% |
| Test (379–743) | TRANSFER | 80,221 | 2,003 | 2.496852% |

The period-wise analysis shows substantial differences in observed fraud prevalence for both CASH_OUT and TRANSFER transactions.

For example:

- CASH_OUT fraud prevalence increased from 0.081846% in validation to 0.645500% in test.
- TRANSFER fraud prevalence increased from 0.337967% in validation to 2.496852% in test.

These observations provide additional context for the overall change in fraud prevalence across chronological periods.

### 5.5 Temporal analysis interpretation

The audit establishes the following descriptive findings:

1. Fraud prevalence is not constant throughout the simulation timeline.
2. The validation and test periods have substantially different observed fraud rates.
3. Fraud prevalence differs across transaction types.
4. The change in observed fraud prevalence is also visible within CASH_OUT and TRANSFER categories.
5. The underlying cause of these differences has not been established.

This analysis indicates temporal distribution differences that warrant further investigation. It does not constitute a formal statistical drift test and does not establish the cause of the observed changes.

## 6. Model and threshold details

| Model | Important configuration | Threshold selection |
|---|---|---|
| Logistic Regression | `class_weight=balanced`; five listed features | Maximum F1 on validation |
| Decision Tree | `max_depth=12`, `min_samples_leaf=100`, `class_weight=balanced`, `random_state=42` | Maximum F1 on validation |
| Random Forest | `n_estimators=100`, `max_depth=16`, `min_samples_leaf=100`, `max_samples=0.5`, `class_weight=balanced_subsample`, `random_state=42` | Maximum F1 on validation |

The modeling datasets retain `step` as the temporal column for chronological partitioning and analysis. It is not included in the five model predictor features listed at the beginning of this report.

## 7. Limitations and follow-up actions

1. **Preserve frozen test evaluations.** Do not tune thresholds, choose features, optimize hyperparameters, or repeatedly select models using the existing test partition.
2. **Maintain comparable evaluation standards.** Logistic Regression currently has validation results only. Do not present its validation metrics as directly comparable with tree-model test metrics.
3. **Investigate temporal distribution differences.** Examine feature distributions and other relevant data characteristics across chronological periods. Use an appropriate future evaluation period for further model development.
4. **Assess operational costs.** Select an operational threshold using documented fraud-loss and investigation-capacity costs, rather than F1 alone, when these business inputs become available.
5. **Check probability calibration.** Evaluate calibration using appropriate evaluation data before relying on predicted probabilities for risk decisions.
6. **Assess temporal stability.** Evaluate performance across suitable future time windows and investigate changes in fraud prevalence and alert volumes.
7. **Complete candidate coverage.** Add other model comparisons only when evaluation artifacts and split/threshold provenance are documented.

The temporal drift audit is descriptive and does not replace formal feature-distribution analysis, statistical drift testing, or future-period model validation.

## 8. Final recommendation

**Current leading experimental candidate: Random Forest.**

Random Forest outperforms Decision Tree on the supplied validation and test metrics. It captures more fraud cases while generating fewer false positives in the supplied test evaluation.

Decision Tree should remain as a simpler benchmark, while Logistic Regression remains the validation-only baseline in this comparison.

The Random Forest result should be treated as a comparative experimental conclusion, not production approval.

Before deployment, further work is required on:

- Temporal stability and future-period validation.
- Probability calibration.
- Operational threshold selection.
- Fraud-loss and investigation-capacity trade-offs.
- Prediction latency and deployment requirements.
- Model monitoring and maintenance.

## 9. Source artifacts

### Model evaluation artifacts

- `reports/paysim/logistic_baseline_metrics.json`
- `reports/paysim/tree_fraud_v1/decision_tree_metrics.json`
- `reports/paysim/tree_fraud_v1/decision_tree_test_metrics.json`
- `reports/paysim/tree_fraud_v1/random_forest_metrics.json`
- `reports/paysim/tree_fraud_v1/random_forest_test_metrics.json`

### Temporal drift audit artifacts

- `reports/paysim/temporal_drift_report.md`
- `reports/paysim/temporal_drift/fraud_by_50_step_interval.csv`
- `reports/paysim/temporal_drift/fraud_by_transaction_type.csv`
- `reports/paysim/temporal_drift/fraud_by_period_and_type.csv`

**Reproducibility note:** The model comparison metrics are retained from the supplied evaluation artifacts. The temporal distribution findings are based on the supplementary audit of the existing PaySim Gold Parquet dataset.

---

**Project:** FINRISK NEXUS
**Dataset:** PaySim
**Target:** `isFraud`
