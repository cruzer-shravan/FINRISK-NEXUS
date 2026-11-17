# FINRISK NEXUS - Home Credit Independent Test Release Record

## Release status

**INDEPENDENT TEST EVALUATION COMPLETED**

Model status: Provisionally approved candidate; independent test evaluation completed.
This record does not constitute production approval or evidence of future-population performance.

## Model identity

- Model: Tuned XGBoost with sigmoid calibration
- Estimator: CalibratedClassifierCV
- Model artifact: reports/home_credit/model_selection_locks/20261107_xgboost_sigmoid/model/calibrated_xgboost_tuned_sigmoid.joblib
- Model SHA256: `21B24D48DD54A03080F30D3421D807D74AE04FD7F734C34DA82D3E59E33D644B`
- Model retrained during test evaluation: NO

## Fixed decision policy

- Threshold: `0.17248130501689904`
- Threshold source: pre-approved validation decision policy
- False-positive cost: 1
- False-negative cost: 5
- Business cost: `FP * 1 + FN * 5`
- Threshold optimized on test: NO

## Test dataset

- Dataset: Home Credit application-only test partition
- Rows: 46,127
- Features: 120
- Defaults: 3,724
- Non-defaults: 42,403
- Default rate: 8.0734%
- Test dataset SHA256: `E40CE61D1DADF3B7977E168457DADCF0CD95471542F0FDCC76A4257B231DE243`

## Independent test metrics

| Metric | Result |
|---|---:|
| ROC-AUC | 0.762773 |
| Average Precision | 0.247060 |
| Brier Score | 0.067655 |
| Log Loss | 0.245370 |
| Accuracy | 0.863399 |
| Precision | 0.260279 |
| Recall | 0.375671 |
| F1 | 0.307506 |
| F2 | 0.345074 |
| ECE | 0.005960 |

## Confusion matrix

| Actual / Predicted | Non-default | Default |
|---|---:|---:|
| Actual non-default | 38,427 (TN) | 3,976 (FP) |
| Actual default | 2,325 (FN) | 1,399 (TP) |

## Business cost

- Validation business cost: 15,595
- Independent test business cost: 15,601
- Difference: 6
- Test business cost per application: 0.338218
- Cost weights are illustrative assumptions, not actual lender loss estimates.

## Validation-to-test assessment

The test results are broadly consistent with validation results. ROC-AUC,
Average Precision, Brier Score and F1 show small differences. This is
encouraging evidence of generalization on this held-out sample, not a guarantee
of future performance or performance across other lending populations.

## Integrity controls

- Locked model hash verified: PASS
- Locked model hash unchanged after inference: PASS
- Test dataset hash unchanged after evaluation: PASS
- Test threshold optimization: NOT PERFORMED
- Model retraining: NOT PERFORMED
- Customer-level prediction file generated: NO

## Release artifacts

- `test_metrics.json`
- `TEST_EVALUATION_REPORT.md`
- `confusion_matrix.csv`
- `calibration_reliability.csv`
- `test_calibration_curve.png`
- `release_artifact_checksums.csv`

The checksum manifest includes the locked model, test partition and evaluation outputs.

## Restrictions and limitations

1. Do not tune the threshold or select another model using this test partition.
2. The cost weights are illustrative and require business validation.
3. The test is a stratified random holdout, not a temporal validation.
4. The evaluation does not establish production readiness, regulatory compliance,
   fairness, or stability under future data drift.
5. Any future model or policy changes require a new development and validation
   cycle and a separately reserved test set.

Record created UTC: 2026-11-07T06:28:02.632829+00:00
