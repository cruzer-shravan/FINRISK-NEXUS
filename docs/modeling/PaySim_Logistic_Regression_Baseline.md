# PaySim Logistic Regression Baseline --- Frozen Record

**Project:** FINRISK NEXUS\
**Experiment ID:** `paysim-logreg-baseline-v1`\
**Status:** Frozen baseline; not a production model\
**Evaluation:** Chronological validation set only\
**Test set:** Not used for training, threshold selection, or this
evaluation

------------------------------------------------------------------------

## 1. Purpose

This document records the first reproducible machine-learning baseline
for PaySim fraud detection. The purpose is to establish a reference
point for later models, including tree-based classifiers.

Do not overwrite this record when running new experiments. Create a new
experiment record for each subsequent model.

## 2. Data and split

Source Gold dataset:

`data/gold/paysim/transaction_features.parquet`

Modeling artifacts:

-   `data/modeling/paysim/train.parquet`
-   `data/modeling/paysim/validation.parquet`
-   `data/modeling/paysim/test.parquet`
-   `data/modeling/paysim/split_manifest.json`

  Split          Simulation steps        Rows   Fraud cases   Fraud rate
  ------------ ------------------ ----------- ------------- ------------
  Train                    1--323   4,463,587         3,643    0.081616%
  Validation             324--378     980,416           564    0.057527%
  Test                   379--743     918,617         4,006    0.436090%

The chronological split was structurally validated. Time ranges do not
overlap and the three splits account for all 6,362,620 source
transactions.

The validation and test distributions differ from the training
distribution. The temporal variation is documented in
`docs/data_engineering/PaySim_Temporal_Audit.md`; its underlying cause
has not been established.

## 3. Model configuration

-   **Estimator:** `sklearn.linear_model.LogisticRegression`
-   **Class weighting:** `balanced`
-   **Solver:** `lbfgs`
-   **Maximum iterations:** `1000`
-   **Random state:** `42`
-   **Target:** `isFraud`
-   **Numeric features:** `amount`, `oldbalanceOrg`, `oldbalanceDest`,
    `log_amount`
-   **Categorical feature:** `type`
-   **Preprocessing:** median imputation and standard scaling for
    numeric columns; most-frequent imputation and one-hot encoding for
    `type`
-   **Pipeline:** preprocessing and estimator are stored together in a
    scikit-learn pipeline

The fitted preprocessing is learned from the training data through the
pipeline. The validation set is used for evaluation and threshold
selection.

## 4. Validation ranking metrics

  Metric                             Result
  ----------------------------- -----------
  Average Precision (PR-AUC)       0.170028
  ROC-AUC                          0.982078
  Validation fraud prevalence     0.057527%
  Validation transactions           980,416
  Validation fraud cases                564

Average Precision is reported as the PR-AUC summary used by
scikit-learn. It should not be confused with ROC-AUC. Given the severe
class imbalance, PR-AUC, precision, recall, and false-alert counts are
central to model assessment.

## 5. Validation decision-threshold results

### Default threshold: 0.50

  Metric                     Result
  ---------------------- ----------
  Precision                0.007726
  Recall                   0.980496
  F1                       0.015332
  True positives                553
  False positives            71,022
  True negatives            908,830
  False negatives                11
  Transactions flagged       71,575

### F1-selected threshold on validation

The threshold was selected by maximizing F1 over the thresholds returned
by `precision_recall_curve` on the validation set.

  Metric                           Result
  ---------------------- ----------------
  Selected threshold       0.999980268813
  Precision                      0.435540
  Recall                         0.221632
  F1                             0.293772
  True positives                      125
  False positives                     162
  True negatives                  979,690
  False negatives                     439
  Transactions flagged                287

The selected threshold is a validation-derived operating point, not a
calibrated probability cutoff and not a production decision. It is
optimized for F1, which does not necessarily reflect operational fraud
costs or investigation capacity.

Threshold diagnostics also showed that at threshold `0.9999`, precision
was approximately 0.281947, recall 0.246454, F1 0.263009, with 139 true
positives, 354 false positives, and 425 false negatives.

## 6. Interpretation

1.  The model achieves high ROC-AUC but much lower Average Precision,
    reflecting the difficulty of finding rare fraud cases.
2.  At threshold 0.50, recall is high but the number of false positives
    is very large.
3.  Raising the threshold reduces alerts and improves precision at the
    cost of missing more fraud cases.
4.  The extremely high F1-selected threshold should not be interpreted
    as a calibrated probability. Class weighting changes the training
    objective and can affect probability interpretation.
5.  The validation period contains fewer fraud cases and a lower fraud
    rate than the test period. Final performance on later simulation
    periods may differ materially.
6.  The model is a baseline for comparison, not a production-ready fraud
    detector.

## 7. Leakage and evaluation controls

-   `isFraud` is the target and is not an input feature.
-   `nameOrig` and `nameDest` are excluded from the baseline.
-   `newbalanceOrig`, `newbalanceDest`, and `isFlaggedFraud` are
    excluded from the baseline.
-   Engineered balance-error and mismatch features are excluded pending
    review of their formulas and availability at scoring time.
-   `step` is used for chronological partitioning but is not a baseline
    predictor.
-   The test dataset was not loaded by the baseline training script and
    must remain untouched until model and threshold selection are
    complete.
-   Do not rebalance validation or test data.

## 8. Frozen artifacts

The following artifacts were created by the baseline workflow:

-   Model pipeline: `models/paysim/logistic_regression_baseline.joblib`
-   Validation metrics: `reports/paysim/logistic_baseline_metrics.json`
-   Training script: `scripts/train_paysim_logistic_baseline.py`
-   Threshold analysis script:
    `scripts/analyze_paysim_baseline_thresholds.py`

The saved metrics JSON and fitted model file should be retained as the
original baseline artifacts. Do not replace them with outputs from later
experiments. Before committing, confirm these files exist and review the
metrics JSON against this record.

## 9. Limitations and next experiment

This experiment has not evaluated probability calibration,
business-cost-sensitive thresholds, or generalization to real-world
fraud. The cause of temporal variation in the synthetic data is
unresolved.

**Next experiment:** train a tree-based model using the same
chronological train/validation splits and feature contract. Record its
results under a separate experiment name. Select models and thresholds
using training/validation data only; use the test set only for the final
evaluation after decisions are fixed.

------------------------------------------------------------------------

**Frozen baseline ID:** `paysim-logreg-baseline-v1`\
**Record date:** 2026-10-19\
**Status note:** Freeze means the experiment record is preserved as a
comparison point; it does not imply production approval.
