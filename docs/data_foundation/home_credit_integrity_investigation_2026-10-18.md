# FINRISK NEXUS — Home Credit Data Integrity Investigation

**Date:** 18 October 2026
**Dataset:** Home Credit Default Risk
**Source:** Kaggle, as confirmed by the project owner
**Status:** Structural checks completed; referential-integrity investigation remains open.

## 1. Objective

The objective of today's work was to investigate unmatched identifiers discovered during Home Credit dataset validation, verify the structural consistency of the relevant CSV files, and decide whether further investigation was necessary before proceeding with feature engineering.

The investigation focused on the relationships between the bureau, previous-application, and associated historical transaction tables.

## 2. Work Completed

### 2.1 Investigated unmatched identifiers

Reviewed the results from the existing Home Credit referential-integrity validation and diagnostic scripts.

Four relationships were found to contain unmatched child identifiers:

| Child table | Parent table | Join key | Unmatched distinct IDs |
|---|---|---|---:|
| `bureau_balance.csv` | `bureau.csv` | `SK_ID_BUREAU` | 43,041 |
| `POS_CASH_balance.csv` | `previous_application.csv` | `SK_ID_PREV` | 37,422 |
| `credit_card_balance.csv` | `previous_application.csv` | `SK_ID_PREV` | 11,372 |
| `installments_payments.csv` | `previous_application.csv` | `SK_ID_PREV` | 38,847 |

These counts were reported by the earlier integrity-checking script. They represent distinct unmatched identifiers, not necessarily the number of affected rows.

A diagnostic script subsequently confirmed that sampled identifiers existed in their respective child tables but were absent from the corresponding parent tables.

**Finding:** The sampled unmatched references are reproducible against the local CSV files. Their underlying cause has not yet been established.

### 2.2 Inspected sample records

Reviewed examples from the four affected relationships.

Examples included:

- `SK_ID_BUREAU = 5001740` in `bureau_balance.csv`.
- `SK_ID_PREV = 1000041` in `POS_CASH_balance.csv`.
- `SK_ID_PREV = 1000096` in `credit_card_balance.csv`.
- `SK_ID_PREV = 1000021` in `installments_payments.csv`.

The child records were present, while the corresponding parent identifiers were not found during the previous diagnostic checks.

No records were deleted, corrected, or modified as part of this investigation.

### 2.3 Checked file sizes and modification times

Executed a PowerShell command using `Get-Item` to inspect the six CSV files.

| File | Size (bytes) |
|---|---:|
| `bureau.csv` | 170,016,717 |
| `bureau_balance.csv` | 3,755,928,899 |
| `previous_application.csv` | 404,973,293 |
| `POS_CASH_balance.csv` | 392,703,158 |
| `credit_card_balance.csv` | 424,582,605 |
| `installments_payments.csv` | 723,118,349 |

The modification timestamps were close together, around 11 October 2026.

**Interpretation:** These checks established the current local file sizes and timestamps. They did not establish file completeness or prove that all files originated from the same original archive.

### 2.4 Created and executed a structural verification script

Created:

` scripts/check_home_credit_file_counts.py `

The script checks:

- Whether each expected CSV file exists.
- The physical line count minus the header.
- The number of columns.
- The column headers.
- Whether the observed row count matches the expected reference count.

Executed:

```powershell
python scripts/check_home_credit_file_counts.py
```

All six files matched the expected reference row and column counts.

| File | Data rows | Columns | Result |
|---|---:|---:|---|
| `bureau.csv` | 1,716,428 | 17 | MATCH |
| `bureau_balance.csv` | 27,299,925 | 3 | MATCH |
| `previous_application.csv` | 1,670,214 | 37 | MATCH |
| `POS_CASH_balance.csv` | 10,001,358 | 8 | MATCH |
| `credit_card_balance.csv` | 3,840,312 | 23 | MATCH |
| `installments_payments.csv` | 13,605,401 | 8 | MATCH |

The displayed column headers were also consistent with the expected Home Credit table schemas.

**Limitation:** The script counts physical lines and assumes one CSV record per line. Matching row counts and headers is a structural check, not proof of complete data integrity.

### 2.5 Corrected the scope of customer-ID validation

An earlier validation compared customer IDs in `bureau.csv` and `previous_application.csv` against only one application table, which could produce misleading unmatched-customer results.

The validation logic was adjusted to compare these IDs against the combined customer-ID set from `application_train.csv` and `application_test.csv`.

The resulting checks reported:

- Zero overlap between train and test customer IDs.
- Zero unmatched distinct customer IDs for the checked `SK_ID_CURR` relationships in `bureau.csv` and `previous_application.csv`.

These results are distinct from the unresolved `SK_ID_BUREAU` and `SK_ID_PREV` relationships listed above.

## 3. Key Findings

1. All six inspected CSV files match the expected reference row and column counts.
2. The displayed headers match the expected Home Credit table structures.
3. The train/test customer-ID overlap check reported zero overlapping IDs.
4. Four child-to-parent identifier relationships contain unmatched distinct IDs.
5. Sample unmatched references were confirmed to exist in child tables but not in their corresponding parent tables.
6. The fact that the files were downloaded from Kaggle has been confirmed by the project owner; the exact original archive and file provenance have not been independently verified.
7. The cause and modelling impact of the unmatched references remain unresolved.

## 4. Scripts and Commands Used

### Existing scripts

- `scripts/profile_home_credit.py` — profiles the Home Credit tables.
- `scripts/validate_home_credit_integrity.py` — checks identifiers and table relationships.
- `scripts/investigate_home_credit_orphans.py` — investigates sample unmatched identifiers.

### Script created today

- `scripts/check_home_credit_file_counts.py` — checks expected row counts, column counts, and headers.

### Command executed successfully

```powershell
python scripts/check_home_credit_file_counts.py
```

A preliminary one-line Python command failed because nested quotation marks were interpreted incorrectly by PowerShell. It did not modify the data. The equivalent check was then implemented as a standalone Python script and executed successfully.

## 5. Decisions Made

To keep the project moving toward its financial-risk modelling objectives:

- Do not delete unmatched records.
- Do not manufacture missing parent records.
- Do not weaken the existing integrity checks merely to obtain a passing result.
- Do not re-download the entire dataset without evidence that the local files are incorrect.
- Avoid further open-ended investigation unless the unresolved relationships materially affect the first modelling task.

The next priority is to determine which Home Credit tables and relationships are required for the initial modelling dataset.

If the unresolved references affect the selected features or analytical joins, investigate them further before modelling. Otherwise, document the limitation and proceed with appropriate validation of the actual modelling dataset.

## 6. Pending Work

- [ ] Record the four unmatched-reference counts in the data-quality documentation.
- [ ] Confirm the exact source archive or download provenance if needed.
- [ ] Determine whether the unmatched relationships affect the first modelling dataset.
- [ ] Investigate the cause further only if it materially affects modelling or data reliability.
- [ ] Begin feature engineering for the selected financial-risk prediction task.
- [ ] Run relevant automated tests when the corresponding implementation is ready.

## 7. Current Project Status

**Home Credit data foundation:** Structural verification completed for the six inspected tables.

**Referential integrity:** Partially validated; four relationships remain unresolved.

**Machine learning:** No ML algorithms have been implemented or evaluated in FINRISK NEXUS as part of this work.

**Automated testing:** The work documented here does not establish that the pytest suite has been executed.

**Next milestone:** Establish the initial modelling dataset and begin feature engineering, with targeted integrity checks for the data used by that task.

---

**Summary:** Today's work verified the structure of six Home Credit CSV files, revisited customer-ID validation, and confirmed that sampled unmatched references exist in the local data. The unresolved relationships are documented rather than silently corrected. The investigation will be bounded by their relevance to the first modelling task.
