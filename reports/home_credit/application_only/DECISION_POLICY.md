# Home Credit – Lending Risk Decision Policy

## 1. Purpose

Document the proposed model-selection and classification
decision policy for the Home Credit application-only model.

## 2. Validation Evidence

Validation dataset: 46,127 rows

Candidate models:
- Tuned Logistic Regression
- Tuned XGBoost Sigmoid
- Tuned LightGBM Sigmoid

## 3. Business Cost Assumptions

False Positive (FP) cost = 1
False Negative (FN) cost = 5

The cost ratio is illustrative and represents a lending
scenario where missing a genuine default is more expensive
than incorrectly flagging a creditworthy applicant.

These are relative cost units, not actual currency.

## 4. Proposed Model Selection

Selected candidate: Tuned XGBoost Sigmoid

Proposed decision threshold: 0.17248130501689904

Validation business cost: 15,595

Selection criterion:
Minimum validation business cost under the High Default
Cost scenario.

## 5. Alternative Model

Tuned LightGBM Sigmoid:

- ROC-AUC: 0.7595155
- Average Precision: 0.2492964
- Brier Score: 0.0677003
- High Default Cost: 15,700

XGBoost achieves a lower business cost by 105 units.

LightGBM has marginally better predictive and calibration
metrics. This trade-off must be acknowledged.

## 6. Decision Status

PROVISIONALLY APPROVED – 2026-11-07.

The model and threshold must not be treated as finally
locked until the business-cost policy is approved.

## 7. Test Evaluation Protocol

The independent test partition must remain untouched
until model selection and threshold selection are finalized.

The selected model and threshold must be evaluated once
on the independent test partition.

No threshold optimization using test data is permitted.

## 8. Reproducibility

Validation evidence freeze:

reports/home_credit/validation_freezes/20261107_113939/

Freeze integrity:
- Total files: 42
- Verified files: 42
- Failed files: 0

The validation freeze is preserved separately from
subsequent decision documentation.

## 9. Approval

Business-cost policy: Provisionally approved

Model selection: Provisionally approved

Threshold selection: Provisionally approved

Independent test evaluation: Not performed
