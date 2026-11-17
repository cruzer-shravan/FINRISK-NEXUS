# FINRISK NEXUS — PaySim Model Release Record

## 1. Release Status

Status: Evaluation Frozen — Release Documentation
Dataset: PaySim
Experiment: paysim-tree-fraud-v1

The PaySim modelling and holdout test evaluations are frozen.
No further model training or test-set tuning is authorised
as part of this release.

## 2. Model Comparison

Three models were evaluated:

- Logistic Regression: baseline
- Decision Tree: benchmark model
- Random Forest: selected experimental champion

Random Forest was selected based on its validation performance
and subsequently demonstrated stronger performance on the
frozen chronological test set.

## 3. Frozen Test Results

| Metric | Decision Tree | Random Forest |
|---|---:|---:|
| Average Precision | 0.813664 | 0.893937 |
| ROC-AUC | 0.987809 | 0.998895 |
| Precision | 0.920015 | 0.951872 |
| Recall | 0.628807 | 0.710934 |
| F1 Score | 0.747034 | 0.813947 |
| True Positives | 2519 | 2848 |
| False Positives | 219 | 144 |
| False Negatives | 1487 | 1158 |

## 4. Selected Model

Model: Random Forest
Artifact: models/paysim/tree_fraud_v1/random_forest.joblib
Experiment: paysim-tree-fraud-v1
Decision Threshold: 0.9305254794485053
Threshold Source: Validation F1 optimisation

The threshold was selected on validation predictions and
kept unchanged during holdout test evaluation.

The model was not retrained during test evaluation.

## 5. Data Partitions

Training:
data/modeling/paysim/train.parquet

Validation:
data/modeling/paysim/validation.parquet

Test:
data/modeling/paysim/test.parquet

The three partition files are preserved independently.

## 6. Environment and Integrity Records

Environment:
reports/paysim/environment_freeze.txt

SHA-256 artifact manifest:
reports/paysim/paysim_artifact_checksums.csv

The checksum manifest contains 14 entries covering the
training/evaluation scripts, model artifacts, evaluation
metrics and modelling datasets.

## 7. Important Evaluation Considerations

The chronological test partition has a fraud prevalence
of approximately 0.4361%, compared with approximately
0.0575% in validation.

This indicates a substantial change in fraud prevalence
between validation and test periods.

Consequently, the observed test precision and other
classification metrics should not be assumed to remain
constant in a live deployment.

The test set must not be used for further threshold
optimisation or model selection.

## 8. Release Decision

Random Forest is the experimental champion for the PaySim
fraud detection pipeline.

The frozen evaluation results are retained as the
benchmark for subsequent integration.

Production approval remains subject to the wider FINRISK
NEXUS integration, operational validation, monitoring,
and deployment checks.

## 9. Next Milestone

Proceed to Home Credit data preparation, feature engineering,
modelling, validation and evaluation.

Do not reopen the frozen PaySim test evaluation without
a separately justified experiment and a new evaluation protocol.
