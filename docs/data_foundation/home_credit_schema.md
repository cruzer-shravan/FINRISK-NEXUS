# FINRISK NEXUS — Home Credit Schema Documentation

**Phase:** 1 — Data Foundation
**Dataset:** Home Credit Default Risk
**Status:** Initial schema exploration completed; full validation pending
**Source:** Kaggle Home Credit Default Risk competition
**Raw data location:** `data/raw/home_credit/`

---

## 1. Purpose

This document describes the structure of the Home Credit Default Risk dataset used in FINRISK NEXUS.

The dataset will support the development of credit-default risk features and predictive models. It contains application-level information and historical credit, repayment, and account-balance records.

The raw dataset must remain unchanged. Data cleaning, transformations, and feature engineering will be implemented in subsequent pipeline stages.

## 2. Dataset inventory

The downloaded dataset contains ten files.

| File | Columns | Purpose |
|---|---:|---|
| `application_train.csv` | 122 | Main training applications and default-risk target |
| `application_test.csv` | 121 | Applications without the target |
| `bureau.csv` | 17 | Previous credits reported by credit bureaus |
| `bureau_balance.csv` | 3 | Monthly status of bureau credit records |
| `previous_application.csv` | 37 | Historical loan applications |
| `POS_CASH_balance.csv` | 8 | Monthly point-of-sale and cash-loan records |
| `credit_card_balance.csv` | 23 | Historical credit-card balances and payments |
| `installments_payments.csv` | 8 | Installment schedules and payment events |
| `HomeCredit_columns_description.csv` | 5 | Data dictionary |
| `sample_submission.csv` | To be verified | Example competition submission format |

Column counts are based on the initial schema inspection. The submission-file column count remains to be verified.

## 3. Primary prediction target

**Target column:** `TARGET`
**Location:** `application_train.csv`

The data dictionary describes the target as an indicator of payment difficulties.

- `TARGET = 1`: The client experienced the specified payment difficulties.
- `TARGET = 0`: The client did not meet the dataset's positive-target definition.

The target must be excluded from the feature matrix during model training.

The test dataset does not contain `TARGET`.

### Initial class-distribution observation

A sample of 1,000 training records contained:

| Target | Records | Proportion |
|---|---:|---:|
| 0 | 930 | 93% |
| 1 | 70 | 7% |

This is an initial sample result, not a full-dataset class-distribution report. Full-dataset class balance must be measured before modeling.

## 4. Table descriptions and identifiers

### 4.1 Application tables

**Files:** `application_train.csv`, `application_test.csv`

Primary identifier: `SK_ID_CURR`

The application tables contain applicant and loan-application attributes, including income, requested credit, demographic information, housing characteristics, and employment-related fields.

Important columns include:

- `SK_ID_CURR`: Application/customer identifier used across related tables.
- `TARGET`: Default-risk target, present only in training data.
- `AMT_INCOME_TOTAL`: Applicant income.
- `AMT_CREDIT`: Credit amount.
- `AMT_ANNUITY`: Loan annuity.
- `DAYS_BIRTH`: Age-related relative-day field.
- `DAYS_EMPLOYED`: Employment-related relative-day field.

The first 1,000 training records contained 122 columns, 65 floating-point columns, 41 integer columns, and 16 string columns.

### 4.2 Bureau records

**File:** `bureau.csv`

Identifiers: `SK_ID_CURR`, `SK_ID_BUREAU`

Contains credit records reported by external credit bureaus.

One application/customer may have multiple bureau records. `SK_ID_BUREAU` identifies a bureau credit record and connects it to `bureau_balance.csv`.

### 4.3 Bureau balance history

**File:** `bureau_balance.csv`

Identifiers: `SK_ID_BUREAU`, `MONTHS_BALANCE`

Contains monthly status observations for bureau credit records.

A bureau credit may have multiple monthly observations. `MONTHS_BALANCE` is a relative month index rather than an absolute calendar date.

### 4.4 Previous applications

**File:** `previous_application.csv`

Identifiers: `SK_ID_CURR`, `SK_ID_PREV`

Contains historical loan applications.

An application/customer may have multiple previous applications. `SK_ID_PREV` connects these records to the relevant historical balance and installment tables.

### 4.5 POS and cash balance

**File:** `POS_CASH_balance.csv`

Identifiers: `SK_ID_PREV`, `SK_ID_CURR`

Contains historical point-of-sale and cash-loan balance records, including monthly information and delinquency-related fields.

### 4.6 Credit-card balance

**File:** `credit_card_balance.csv`

Identifiers: `SK_ID_PREV`, `SK_ID_CURR`

Contains credit-card balance, credit-limit, drawing, and payment information over time.

### 4.7 Installment payments

**File:** `installments_payments.csv`

Identifiers and relevant fields:

- `SK_ID_PREV`
- `NUM_INSTALMENT_VERSION`
- `NUM_INSTALMENT_NUMBER`
- `DAYS_INSTALMENT`
- `DAYS_ENTRY_PAYMENT`
- `AMT_INSTALMENT`
- `AMT_PAYMENT`

This table records installment schedules and payment events. Multiple rows can share the same previous-loan and installment identifiers.

An initial inspection of the complete file loaded 13,605,401 records. The selected combination of `SK_ID_PREV`, `NUM_INSTALMENT_VERSION`, and `NUM_INSTALMENT_NUMBER` produced repeated key combinations.

These repetitions must not automatically be treated as erroneous duplicates. Payment events and amounts must be preserved until their meaning is assessed.

## 5. Conceptual relationship map

The intended relationships are:

- Application tables → bureau records through `SK_ID_CURR`.
- Bureau records → bureau balance history through `SK_ID_BUREAU`.
- Application tables → previous applications through `SK_ID_CURR`.
- Previous applications → POS/cash, credit-card, and installment records through `SK_ID_PREV`.

The historical balance and installment tables also contain `SK_ID_CURR`.

These relationships are based on observed column names and dataset structure. Full-dataset referential-integrity, cardinality, and unmatched-key checks remain pending.

## 6. Temporal fields

| Field | Interpretation |
|---|---|
| `DAYS_BIRTH` | Applicant age expressed in relative days |
| `DAYS_EMPLOYED` | Employment start timing relative to application |
| `DAYS_REGISTRATION` | Registration timing relative to application |
| `DAYS_ID_PUBLISH` | Identity-document timing relative to application |
| `DAYS_LAST_PHONE_CHANGE` | Time since the last phone change |
| `WEEKDAY_APPR_PROCESS_START` | Weekday of application processing |
| `AMT_REQ_CREDIT_BUREAU_DAY` | Number of bureau enquiries in the defined day-level window |

The application table uses relative-day fields, not necessarily absolute calendar dates.

### Special value

`DAYS_EMPLOYED = 365243` occurs repeatedly and is not a realistic employment duration.

In the first 1,000 training records, this value appeared 158 times. It must be treated as a special sentinel value and investigated during preprocessing.

The original values must remain unchanged in the raw dataset.

Temporal features must be checked for leakage and interpreted according to their definitions before use in modeling.

## 7. Initial data-quality findings

The initial exploration identified:

1. Missingness of approximately 65–70% in several housing-related columns in a 1,000-row sample.
2. A 7% positive-target proportion in that sample.
3. A special value of `365243` in `DAYS_EMPLOYED`.
4. Multiple payment-event records sharing installment key combinations.
5. Relational identifiers that appear consistent in sampled records.

These findings are preliminary. Missingness, class distribution, key uniqueness, duplicate records, and referential integrity must be assessed on the full datasets before modeling.

## 8. Data handling principles

- Preserve all downloaded source files under `data/raw/home_credit/`.
- Do not overwrite, clean, or rename original CSV files.
- Do not assume repeated business keys indicate duplicate rows.
- Keep raw datasets out of GitHub.
- Perform transformations in separate processing stages.
- Record schema changes and data-quality decisions in project documentation.
- Validate historical data availability before constructing predictive features.

## 9. Outstanding validation tasks

- [ ] Verify the complete column descriptions for all tables.
- [ ] Measure full-dataset row counts and target distribution.
- [ ] Validate key uniqueness and cardinality across complete tables.
- [ ] Check unmatched foreign keys and orphaned historical records.
- [ ] Audit exact duplicate rows separately from repeated business keys.
- [ ] Document missing-value percentages for the complete tables.
- [ ] Identify and document sentinel values and special encodings.
- [ ] Complete a leakage review before feature engineering.
- [ ] Create a machine-readable schema and automated validation tests.

## 10. References

- Home Credit Default Risk competition: https://www.kaggle.com/competitions/home-credit-default-risk/data
- Local data dictionary: `data/raw/home_credit/HomeCredit_columns_description.csv`

---

**Document status:** Initial schema exploration recorded. Full-dataset validation and production schema approval are pending.
