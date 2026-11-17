# Home Credit — Independent Test Evaluation

## Evaluation status

**COMPLETED**

This is the first independent evaluation of the provisionally approved,
locked model on the untouched test partition.

## Model and decision policy

- Model: Tuned XGBoost with sigmoid calibration
- Model artifact: `reports\home_credit\model_selection_locks\20261107_xgboost_sigmoid\model\calibrated_xgboost_tuned_sigmoid.joblib`
- Model SHA256: `21B24D48DD54A03080F30D3421D807D74AE04FD7F734C34DA82D3E59E33D644B`
- Fixed decision threshold: `0.17248130501689904`
- Threshold source: pre-approved validation decision policy
- False-positive cost: 1
- False-negative cost: 5
- Test threshold optimization: **Not performed**
- Model retraining: **Not performed**

## Test partition

- Rows: 46,127
- Features: 120
- Actual defaults: 3,724
- Actual non-defaults: 42,403
- Actual default rate: 8.0734%
- Test dataset SHA256: `e40ce61d1dadf3b7977e168457dadcf0cd95471542f0fdcc76a4257b231de243`

## Classification performance

| Metric | Test result |
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

## Confusion matrix

| | Predicted non-default | Predicted default |
|---|---:|---:|
| Actual non-default | 38,427 (TN) | 3,976 (FP) |
| Actual default | 2,325 (FN) | 1,399 (TP) |

## Business-cost evaluation

Formula: `FP × 1 + FN × 5`

- False positives: 3,976
- False negatives: 2,325
- Total test business cost: **15,601**
- Cost per application: 0.338218

These cost weights are the illustrative assumptions approved for this
evaluation; they are not actual lender loss estimates.

## Calibration diagnostics

- Quantile bins requested: 10
- Actual bins: 10
- Expected Calibration Error: 0.005960

See `calibration_reliability.csv` and `test_calibration_curve.png`.

## Integrity and evaluation controls

- Locked model hash verified before evaluation: PASS
- Locked model hash unchanged after evaluation: PASS
- Test partition hash unchanged after evaluation: PASS
- Threshold optimized on test: NO
- Model retrained: NO
- Customer-level prediction file generated: NO

## Interpretation restriction

Test results are an independent estimate of generalization performance.
They must not be used to tune the threshold or select a replacement model.
Any future model or policy change requires a new development and validation
cycle with a separate untouched test set.
