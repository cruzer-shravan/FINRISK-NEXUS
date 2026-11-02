# PaySim Temporal Audit

**Project:** FINRISK NEXUS\
**Dataset:** PaySim --- Gold transaction features\
**Audit status:** Completed --- findings documented; temporal shift
requires further investigation\
**Purpose:** Record the observed time-dependent transaction and fraud
patterns before baseline model training.

------------------------------------------------------------------------

## 1. Scope and source

This audit uses the validated Gold-layer dataset:

`data/gold/paysim/transaction_features.parquet`

The source contains **6,362,620 transactions**, spanning simulation
steps **1--743**, with **8,213 fraud cases**. The dataset has 16
columns, no reported missing values, and no exact duplicate rows from
the prior inspection.

The audit examines: - Transaction volumes and fraud rates over time. -
Fraud distribution by transaction type. - Differences in fraud rates
across the chronological train, validation, and test periods.

This document records observed patterns. It does not establish the cause
of those patterns.

## 2. Overall fraud prevalence

  Measure                         Result
  -------------------------- -----------
  Total transactions           6,362,620
  Fraud cases                      8,213
  Non-fraud cases              6,354,407
  Overall fraud prevalence       0.1291%
  Simulation steps                1--743

Fraud is a highly imbalanced target. A classifier that predicts every
transaction as non-fraud would achieve approximately **99.87% accuracy**
while detecting no fraud. Accuracy must therefore not be used as the
sole model-selection metric.

Recommended evaluation measures include precision, recall, F1,
precision-recall AUC (PR-AUC), confusion matrices, and calibration where
appropriate.

## 3. Fraud by transaction type

  Transaction type      Transactions   Fraud cases    Fraud rate
  ------------------ --------------- ------------- -------------
  TRANSFER                   532,909         4,097       0.7688%
  CASH_OUT                 2,237,500         4,116       0.1840%
  CASH_IN                  1,399,284             0       0.0000%
  DEBIT                      414,332             0       0.0000%
  PAYMENT                  2,151,495             0       0.0000%
  **Total**            **6,362,620**     **8,213**   **0.1291%**

All recorded fraud cases in this dataset occur in `TRANSFER` and
`CASH_OUT`. The `type` feature is therefore a relevant baseline
predictor.

**Interpretation caution:** This is a property of the synthetic PaySim
data used in this project. It must not be generalized to real-world
payment fraud without independent evidence.

## 4. Fraud prevalence over time

The audit grouped the simulation into 50-step intervals. Results from
the terminal audit are:

    First step in interval   Transactions   Fraud cases   Fraud rate
  ------------------------ -------------- ------------- ------------
                         1      1,030,263           590      0.0573%
                        51         29,253           600      2.0511%
                       101        451,852           520      0.1151%
                       151        870,814           544      0.0625%
                       201        866,110           587      0.0678%
                       251        825,358           560      0.0678%
                       301        870,265           500      0.0575%
                       351        843,115           576      0.0683%
                       401        227,687           580      0.2547%
                       451         47,090           504      1.0703%
                       501        104,212           504      0.4836%
                       551         93,028           548      0.5891%
                       601         22,724           578      2.5416%
                       651         69,410           532      0.7665%
                       701         11,421           490      4.2903%

The distribution is uneven: transaction volume changes substantially
across intervals, and some intervals have much higher fraud rates than
others. The final interval (steps 701--743) contains 490 fraud cases
among 11,421 transactions, a fraud rate of 4.2903%.

The earlier coarse summary showed rates rising in later periods, but the
50-step view demonstrates that the trend is **not a smooth, monotonic
increase**. Some intervals have sharp spikes followed by lower rates.
This should be described as temporal variation and a possible
distribution shift, not as a proven single-direction trend.

## 5. Chronological split record

The modeling preparation script produced these non-overlapping periods:

  Split          Step range    Transactions   Fraud cases    Fraud rate
  ------------ ------------ --------------- ------------- -------------
  Train              1--323       4,463,587         3,643     0.081616%
  Validation       324--378         980,416           564     0.057527%
  Test             379--743         918,617         4,006     0.436090%
  **Total**      **1--743**   **6,362,620**     **8,213**   **0.1291%**

Structural validation passed: - The time ranges do not overlap. - All
three splits have matching columns. - Combined row count equals the
source row count. - All rows are accounted for across the three splits.

The test fraud rate is approximately 5.35 times the training fraud rate
(0.436090% versus 0.081616%). This difference warrants investigation and
must be disclosed when reporting test performance.

## 6. Fraud by transaction type across the splits

The audit reported the following values:

  Period                        Type         Transactions   Fraud cases   Fraud rate
  ----------------------------- ---------- -------------- ------------- ------------
  Early (steps 1--323)          CASH_IN           977,376             0    0.000000%
  Early (steps 1--323)          CASH_OUT        1,582,647         1,831      0.1157%
  Early (steps 1--323)          DEBIT              27,764             0    0.000000%
  Early (steps 1--323)          PAYMENT         1,506,552             0    0.000000%
  Early (steps 1--323)          TRANSFER          369,248         1,812      0.4907%
  Validation (steps 324--378)   CASH_IN           217,839             0    0.000000%
  Validation (steps 324--378)   CASH_OUT          344,551           282      0.0818%
  Validation (steps 324--378)   DEBIT               7,080             0    0.000000%
  Validation (steps 324--378)   PAYMENT           327,506             0    0.000000%
  Validation (steps 324--378)   TRANSFER           83,440           282      0.3380%
  Test (steps 379--743)         CASH_IN           204,069             0    0.000000%
  Test (steps 379--743)         CASH_OUT          310,302         2,003      0.6455%
  Test (steps 379--743)         DEBIT               6,588             0    0.000000%
  Test (steps 379--743)         PAYMENT           317,437             0    0.000000%
  Test (steps 379--743)         TRANSFER           80,221         2,003      2.4969%

The later test period has higher observed fraud rates for both
`CASH_OUT` and `TRANSFER`. This supports examining temporal differences
within transaction types, not only differences in the overall type mix.

## 7. Modeling and evaluation decisions

1.  **Primary evaluation:** retain the chronological
    train/validation/test split.
2.  **Do not rebalance or resample the validation or test sets.**
    Preserve their observed distributions for evaluation.
3.  **Model selection:** use training data to fit models and validation
    data to compare models and choose a decision threshold.
4.  **Final test:** use the test set only after model and threshold
    choices are fixed.
5.  **Baseline feature set:** begin with `type`, `amount`,
    `oldbalanceOrg`, `oldbalanceDest`, and `log_amount`. Retain `step`
    for splitting and temporal analysis, but omit it from the first
    baseline feature set. Evaluate its predictive value separately
    later.
6.  **Target:** `isFraud`.
7.  **Leakage precautions:** exclude post-transaction balances and the
    existing `isFlaggedFraud` rule-based flag from the initial baseline.
    Review the formulas and scoring-time availability of engineered
    balance-error and mismatch features before considering them.
8.  **Metrics:** report precision, recall, F1, PR-AUC, confusion matrix,
    and calibration measures where suitable. Accuracy may be reported as
    a secondary metric only.
9.  **Secondary experiment:** a stratified random split may be used for
    comparison, but it must be clearly labelled as secondary and not
    replace the primary chronological evaluation.

## 8. Limitations and open questions

-   The dataset is synthetic; observed patterns may reflect the
    simulation design.
-   The cause of the temporal variation has not been established.
-   Transaction counts vary dramatically by simulation step, so both
    counts and rates should be examined.
-   A chronological split measures performance on later simulated
    periods, but it does not by itself prove real-world generalization.
-   The audit is descriptive. It does not establish that any feature is
    causal or that a model will perform well.

## 9. Status and next step

**Status:** Temporal audit completed and structural split validation
passed. The temporal distribution shift is documented; its cause remains
unconfirmed.

**Next step:** Train a reproducible Logistic Regression baseline using
the approved initial features. Evaluate and tune only on the training
and validation sets. Keep the chronological test set untouched until
final evaluation.

------------------------------------------------------------------------

**Related artifacts** - Gold dataset:
`data/gold/paysim/transaction_features.parquet` - Modeling splits:
`data/modeling/paysim/train.parquet`, `validation.parquet`,
`test.parquet` - Split manifest:
`data/modeling/paysim/split_manifest.json` - Temporal audit script:
`scripts/audit_paysim_temporal_drift.py` - Split validation script:
`scripts/validate_paysim_modeling_splits.py`
