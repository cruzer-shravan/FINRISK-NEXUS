# PaySim Tree-Based Fraud Modeling â€” Experiment Configuration

**Project:** FINRISK NEXUS
**Dataset:** PaySim
**Milestone:** Tree-Based Fraud Classification
**Experiment ID:** `paysim-tree-fraud-v1`
**Status:** Draft â€” pending artifact validation and Git commit
**Purpose:** Establish a consistent, reproducible configuration for training, evaluating, and comparing tree-based fraud-classification models.

---

## 1. Objective

The objective of this experiment is to evaluate tree-based machine-learning models for identifying fraudulent transactions in the PaySim dataset.

The experiment will build on the existing logistic-regression baseline and assess whether tree-based models improve fraud detection while maintaining a disciplined evaluation methodology.

The experiment must:

- Use the same approved dataset and data partitions for every candidate model.
- Prevent data leakage between training, validation, and test datasets.
- Select decision thresholds using validation data only.
- Evaluate the final selected model on the test set without using test results to tune the model.
- Record model parameters, evaluation metrics, and experiment outputs.
- Preserve the existing logistic-regression baseline for comparison.

## 2. Dataset and Artifact Locations

### 2.1 Gold dataset

Expected dataset path:

`data/gold/paysim/transaction_features.parquet`

The Gold dataset should contain the engineered transaction features approved for modeling.

**Validation required:** Confirm that the file exists and that its schema matches the feature definitions used by the current data-engineering pipeline.

### 2.2 Modeling partitions

Expected partition paths:

- `data/modeling/paysim/train.parquet`
- `data/modeling/paysim/validation.parquet`
- `data/modeling/paysim/test.parquet`

Expected split manifest:

`data/modeling/paysim/split_manifest.json`

The split manifest should record the partitioning method, step ranges, row counts, target counts, and other relevant metadata.

**Validation required:** Confirm that these files exist and that the manifest agrees with the actual partitions before training.

## 3. Target Variable

The intended target variable is:

`isFraud`

Interpretation:

- `1`: Fraudulent transaction
- `0`: Non-fraudulent transaction

The target must not be included among the model's input features.

Before training, verify that the target contains valid binary values and that its distribution is recorded separately for each partition.

## 4. Data Splitting Strategy

The intended strategy is chronological partitioning using the PaySim `step` field.

The `step` field should be used to establish temporal ordering and create the partitions. It should not automatically be included as a predictive feature.

The previously documented split configuration is:

| Partition | Step range | Rows | Fraud cases | Fraud prevalence |
|---|---:|---:|---:|---:|
| Training | 1â€“323 | 4,463,587 | 3,643 | 0.081616% |
| Validation | 324â€“378 | 980,416 | 564 | 0.057527% |
| Test | 379â€“743 | 918,617 | 4,006 | 0.436090% |
| **Total** | **1â€“743** | **6,362,620** | **8,213** | **0.129109%** |

**Important:** These figures are carried forward from the earlier experiment notes and have not been independently revalidated in this document. Confirm them against `split_manifest.json` and the actual Parquet files before treating them as final.

### Data-splitting rules

1. Preserve chronological ordering.
2. Keep the training, validation, and test partitions separate.
3. Fit preprocessing components on training data only.
4. Use validation data for model selection and threshold selection.
5. Reserve test data for final evaluation.
6. Do not change the partitions to improve reported performance.
7. Document any temporal distribution shift observed between partitions.

## 5. Feature Governance

The following feature configuration was proposed for the initial experiment.

### 5.1 Candidate input features

| Feature | Intended treatment |
|---|---|
| `amount` | Numeric transaction amount |
| `oldbalanceOrg` | Numeric origin-account balance before the transaction |
| `oldbalanceDest` | Numeric destination-account balance before the transaction |
| `log_amount` | Engineered transaction-amount feature |
| `type` | Categorical transaction type |

**Validation required:** Confirm that all five columns exist in the Gold dataset and that each feature is available at prediction time. Confirm the definition and transformation used for `log_amount`.

### 5.2 Excluded features

The following columns were identified for exclusion from the proposed initial feature set:

- `step` â€” temporal partitioning field; exclude from predictors unless a separate, explicitly justified experiment evaluates its predictive use.
- `nameOrig` â€” origin-account identifier.
- `nameDest` â€” destination-account identifier.
- `newbalanceOrig` â€” post-transaction origin-account balance.
- `newbalanceDest` â€” post-transaction destination-account balance.
- `isFlaggedFraud` â€” existing flag that may introduce target leakage or encode a separate detection rule.
- `origin_balance_error`
- `destination_balance_error`
- `origin_balance_mismatch`
- `destination_balance_mismatch`

The last four fields must be reviewed against their actual definitions and availability at inference time before the exclusions are finalized.

These exclusions are proposed experimental controls, not a substitute for reviewing the Gold schema and feature-engineering code.

### 5.3 Feature validation requirements

Before fitting any model:

1. Verify the approved feature list against the actual training schema.
2. Confirm that the target and excluded columns are not accidentally passed to the estimator.
3. Check for missing values and unexpected data types.
4. Verify that derived features use information available at prediction time.
5. Record the final feature list in the experiment report.

## 6. Preprocessing Strategy

Preprocessing must be fitted on the training partition only.

### Numeric features

- Inspect missing-value rates.
- If required, use a median imputer fitted on training data.
- Apply the fitted imputer consistently to validation and test data.

### Categorical features

- Encode `type` using a training-fitted encoder.
- Ensure the selected encoding can handle previously unseen categories without failing inference.
- Do not fit or refit the encoder on validation or test data.

### Scaling

Feature scaling is not required by default for tree-based models. Any scaling step must have a clear model-specific justification.

The complete preprocessing and estimator pipeline should be saved together where practical so that training and inference use identical transformations.

## 7. Reproducibility

The proposed random seed is:

`42`

Apply this seed to model components that support it.

For each experiment, record:

- Experiment ID
- Dataset and partition paths
- Target and final feature list
- Random seed
- Preprocessing configuration
- Model class and hyperparameters
- Software versions
- Decision-threshold selection method
- Evaluation metrics
- Output artifact paths

A fixed random seed supports reproducibility but does not guarantee bit-for-bit identical results across every software version or execution environment.

## 8. Baseline Comparison

The tree-based experiments should be compared with the existing logistic-regression baseline.

The following baseline artifacts were previously identified:

- `models/paysim/logistic_regression_baseline.joblib`
- `reports/paysim/logistic_baseline_metrics.json`
- `scripts/train_paysim_logistic_baseline.py`
- `docs/modeling/PaySim_Logistic_Regression_Baseline.md`

**Validation required:** Confirm that these files exist, that the baseline was trained using the same approved partitions, and that its evaluation metrics were generated using the same target definition.

The baseline should remain unchanged during the tree-model comparison. Any baseline retraining should be documented as a separate experiment.

## 9. Model Candidates

Candidate models may include:

1. Decision Tree Classifier
2. Random Forest Classifier
3. Extra Trees Classifier
4. Gradient Boosting Classifier
5. HistGradientBoosting Classifier
6. XGBoost Classifier, if available in the project environment
7. LightGBM Classifier, if available in the project environment
8. CatBoost Classifier, if available in the project environment

The final candidate list must reflect the libraries installed and supported by the project.

**Hyperparameter rule:** Model-specific hyperparameters must be documented before the corresponding experiment is run. Do not claim that a configuration is frozen while candidate parameters remain undecided.

## 10. Evaluation Metrics

Because fraud cases are rare, accuracy alone is not an adequate model-selection metric.

Record the following metrics for each candidate:

- Precision
- Recall
- F1-score
- ROC-AUC
- Average precision / PR-AUC
- Confusion matrix
- Brier score, where meaningful and calculated from predicted probabilities
- Selected decision threshold

Report the fraud prevalence of each partition alongside the metrics.

Use consistent metric definitions and the same evaluation partitions for every candidate.

## 11. Decision-Threshold Policy

The classification threshold must be selected using validation data only.

Proposed initial policy:

1. Generate fraud probabilities on the validation partition.
2. Evaluate candidate thresholds using validation predictions.
3. Select the threshold that maximizes validation F1-score.
4. Document the selected threshold and its validation metrics.
5. Apply the selected threshold unchanged to the test predictions.
6. Report final test metrics without further threshold adjustment.

If multiple thresholds produce equivalent F1-scores, document the tie-breaking rule before running the experiment.

The previously recorded logistic-regression threshold was approximately `0.999980268813`. This value must be verified against the baseline artifact and must not be reused automatically for tree-based models.

## 12. Class Imbalance

Fraud prevalence is expected to be low. The experiment must therefore consider the effects of class imbalance on precision, recall, and threshold selection.

Possible approaches include:

- Class weighting, where supported.
- Model-specific imbalance parameters, where appropriate.
- Threshold optimization on validation predictions.

Any imbalance treatment must be recorded per experiment. It must be fitted or configured without using test outcomes.

Do not apply resampling or synthetic-data generation to validation or test partitions.

## 13. Output Locations

Proposed model output directory:

`models/paysim/tree_fraud_v1/`

Proposed report output directory:

`reports/paysim/tree_fraud_v1/`

Expected outputs should include, as applicable:

- Trained model and preprocessing artifacts
- Model configuration
- Validation metrics
- Final test metrics
- Confusion matrix
- Precision-recall curve
- ROC curve
- Feature-importance results
- Experiment comparison table
- Run metadata and software versions

These directories and filenames are proposed conventions. Confirm the existing repository structure before creating or overwriting artifacts.

## 14. Temporal Distribution Shift

The previously documented partition statistics indicate that fraud prevalence differs between training, validation, and test data.

This difference must be investigated and reported rather than eliminated by changing the chronological split solely to improve performance.

The final report should discuss how this distribution shift may affect precision, recall, calibration, and operational fraud-detection performance.

## 15. Experiment Acceptance Criteria

The experiment is considered complete when:

- [ ] The Gold dataset schema has been verified.
- [ ] The partition files and split manifest have been verified.
- [ ] Row counts and fraud counts have been reconciled.
- [ ] The final feature list and exclusions have been reviewed.
- [ ] Candidate models and hyperparameters have been documented.
- [ ] Preprocessing is fitted on training data only.
- [ ] The logistic-regression baseline is available and comparable.
- [ ] Threshold selection uses validation data only.
- [ ] All candidate models use consistent evaluation partitions.
- [ ] Test metrics are generated only after model selection.
- [ ] Results, configuration, and artifacts are documented.
- [ ] Automated checks and repository validation pass.
- [ ] The configuration is committed to Git and its commit hash is recorded.

## 16. Freeze and Change-Control Policy

This document defines the proposed configuration for the `paysim-tree-fraud-v1` experiment.

The configuration must not be described as frozen until the dataset, partitions, feature definitions, baseline artifacts, candidate hyperparameters, and evaluation procedures have been verified.

After the configuration is frozen:

- Changes to the dataset, split, features, preprocessing, or threshold policy must be documented.
- Material changes should receive a new experiment version or ID.
- Previous results and artifacts must be retained for comparison.
- Test-set results must not be used to make unreported tuning decisions.

### Final status

**Current status:** Draft â€” pending artifact validation and Git commit.

**Freeze confirmation:** Pending.

**Git commit hash:** To be recorded after the configuration is committed successfully.
