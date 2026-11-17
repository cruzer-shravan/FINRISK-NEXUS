# Home Credit – Model Selection Lock

## Approval

Status: PROVISIONALLY APPROVED

Approval Date: 2026-11-07 11:43:03

Lock ID: 20261107_xgboost_sigmoid

## Selected Model

Model: Tuned XGBoost Sigmoid

Artifact:
calibrated_xgboost_tuned_sigmoid.joblib

Calibration Method: Sigmoid

## Decision Policy

False Positive Cost: 1

False Negative Cost: 5

Decision Threshold: 0.172481305016899

Validation Business Cost: 15595

Selection Criterion:
Minimum validation business cost under the
High Default Cost (5:1) scenario.

## Model Integrity

Source SHA256:
21B24D48DD54A03080F30D3421D807D74AE04FD7F734C34DA82D3E59E33D644B

Locked Artifact SHA256:
21B24D48DD54A03080F30D3421D807D74AE04FD7F734C34DA82D3E59E33D644B

Integrity Verification: PASSED

## Validation Evidence

Freeze ID: 20261107_113939

Validation Rows: 46127

## Test Evaluation

Status: NOT PERFORMED

Test partition must remain untouched until
the independent evaluation stage.

Threshold must not be optimized using test data.

## Approval Qualification

This model-selection decision is provisionally
approved for the current project evaluation.

It may be reconsidered if subsequent evidence
or business requirements justify a change.
