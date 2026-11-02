# FINRISK NEXUS --- PaySim Silver Layer Validation Freeze

**Project:** FINRISK NEXUS\
**Dataset:** PaySim transactions\
**Layer:** Silver\
**Status:** Validation passed; freeze record prepared\
**Validation evidence timestamp:** 2026-10-19 (UTC; see the generated
JSON reports for exact timestamps)\
**Scope:** Structural, schema, row-count, and recorded data-quality
checks

------------------------------------------------------------------------

## 1. Purpose

This document records the validation evidence for the PaySim transaction
dataset transformed from the Bronze Parquet source into the Silver
Parquet dataset. It establishes a baseline for downstream feature
engineering and modelling.

A validation freeze records the state that was checked. It does not, by
itself, make files read-only or prevent later modification. The artifact
hashes should be verified and retained as part of the freeze.

## 2. Dataset lineage

  ---------------------------------------------------------------------------------------------------
  Item                                Path
  ----------------------------------- ---------------------------------------------------------------
  Original CSV                        `data/raw/paysim/PS_20174392719_1491204439457_log.csv`

  Bronze source                       `data/bronze/paysim/PS_20174392719_1491204439457_log.parquet`

  Silver output                       `data/silver/paysim/transactions.parquet`

  Silver metadata                     `data/silver/paysim/transactions.metadata.json`

  Data-quality report                 `data/silver/paysim/data_quality_report.json`
  ---------------------------------------------------------------------------------------------------

The Bronze conversion recorded 6,362,620 rows and 11 columns. The Silver
pipeline completed successfully and reported the same row count.

## 3. Validation summary

  ------------------------------------------------------------------------------------
  Check                   Observed result                      Status
  ----------------------- ------------------------------------ -----------------------
  Silver pipeline         `PAYSIM SILVER PIPELINE COMPLETED`   Pass
  execution

  Silver row count        6,362,620                            Pass

  Bronze source row count 6,362,620                            Pass
  in Silver metadata

  Silver row count in     6,362,620                            Pass
  metadata

  Column count            11 source columns and 11 Silver      Pass
                          columns

  Column names and order  Matched according to metadata        Pass

  Schema persisted in     Independently inspected with Polars  Pass
  Parquet

  Recorded quality checks All listed checks have value `0`     Pass

  Metadata validation     `passed`                             Pass
  status

  Data-quality report     `passed`                             Pass
  validation status
  ------------------------------------------------------------------------------------

**Interpretation:** the recorded checks passed. This is not a claim that
every possible data-quality, business-semantic, or modelling-leakage
issue has been ruled out.

## 4. Silver schema

The following types were observed in the persisted Silver Parquet file.

  -----------------------------------------------------------------------
  Column                  Type                    Role / interpretation
  ----------------------- ----------------------- -----------------------
  `step`                  `Int64`                 Simulation time step

  `type`                  `String`                Transaction type

  `amount`                `Float64`               Transaction amount

  `nameOrig`              `String`                Origin account
                                                  identifier

  `oldbalanceOrg`         `Float64`               Origin balance before
                                                  transaction

  `newbalanceOrig`        `Float64`               Origin balance after
                                                  transaction

  `nameDest`              `String`                Destination account
                                                  identifier

  `oldbalanceDest`        `Float64`               Destination balance
                                                  before transaction

  `newbalanceDest`        `Float64`               Destination balance
                                                  after transaction

  `isFraud`               `Int8`                  Fraud label

  `isFlaggedFraud`        `Int8`                  Dataset-provided
                                                  flagged-fraud indicator
  -----------------------------------------------------------------------

Identifiers remain strings so they are not accidentally treated as
quantities. Monetary and balance columns are numeric. The two indicator
columns are integer flags.

## 5. Recorded data-quality checks

The report at `data/silver/paysim/data_quality_report.json` records zero
for each of the following checks:

-   Invalid step
-   Null transaction type
-   Null origin identifier
-   Null destination identifier
-   Invalid amount
-   Invalid origin balance before transaction
-   Invalid origin balance after transaction
-   Invalid destination balance before transaction
-   Invalid destination balance after transaction
-   Invalid `isFraud` value
-   Invalid `isFlaggedFraud` value
-   Negative step
-   Negative amount
-   Negative origin balance before transaction
-   Negative origin balance after transaction
-   Negative destination balance before transaction
-   Negative destination balance after transaction
-   Unexpected transaction type
-   Invalid fraud flag
-   Invalid flagged-fraud flag

The report records `rows_checked: 6362620` and
`validation_status: passed`.

These results mean no violations were recorded by these specific checks.
Additional business-rule checks may still be required during
exploration.

## 6. Fraud-label distribution

The independent Polars inspection returned the following label counts.

    `isFraud` Meaning         Transactions   Percentage
  ----------- ------------ --------------- ------------
            0 Legitimate         6,354,407     99.8709%
            1 Fraudulent             8,213      0.1291%
    **Total**                **6,362,620**     **100%**

The dataset is highly imbalanced. Accuracy must not be used as the sole
model-selection metric. Subsequent modelling should consider precision,
recall, PR-AUC, F1-score, and the operational alert-volume constraint.

The label distribution is a descriptive check, not evidence that the
model can detect fraud.

## 7. Metadata and lineage evidence

The Silver metadata records:

-   Dataset layer: `silver`
-   Dataset: `PaySim transactions`
-   Transformation version: `1.0.0`
-   Format: Parquet
-   Compression: Zstandard (`zstd`)
-   Source and Silver row counts: 6,362,620 each
-   Source and Silver column counts: 11 each
-   `row_counts_match: true`
-   `validation_status: passed`
-   Source SHA-256 and Silver Parquet SHA-256 fields
-   Link to the data-quality report

The metadata shown during review recorded the source hash as:

`d64b01c0629563eca705cf2e47aef51c7fa3d396e6cf2f38f1ba030c44681791`

The Silver hash recorded in that metadata was:

`348c61dbec9ae5937a4522ae0dbb20722d9dec0166520f8becfec33797d8a121`

These are transcribed from the displayed metadata. Recompute the local
file hashes during the freeze procedure below before treating the hashes
as a final immutable baseline.


## 8. Artifact Integrity and SHA-256 Freeze Record

**Freeze date:** 19 October 2026
**Dataset:** PaySim transactions
**Layer:** Silver
**Validation status:** Passed

### 8.1 Frozen artifacts

| Artifact | Location |
|---|---|
| Silver transaction dataset | `data/silver/paysim/transactions.parquet` |
| Silver metadata | `data/silver/paysim/transactions.metadata.json` |
| Data quality report | `data/silver/paysim/data_quality_report.json` |

### 8.2 SHA-256 checksums

| Artifact | SHA-256 |
|---|---|
| `transactions.parquet` | `348C61DBEC9AE5937A4522AE0DBB20722D9DEC0166520F8BECFEC33797D8A121` |
| `transactions.metadata.json` | `DB62DAE725CF34DB4C4A956442DFC37FF1B0CF17ED5CC5EB6CD6A8CF2672455F` |
| `data_quality_report.json` | `4542A41E9736A1442D6B415E524AB71AF2AFF145B6BD926A926D6AC09F6CBFF5` |

### 8.3 Freeze decision

The PaySim Silver pipeline completed successfully. The transaction dataset contains 6,362,620 rows and 11 columns. The recorded quality checks reported zero violations across the implemented validation checks, and the validation status was `passed`.

The SHA-256 checksums establish the integrity reference for the three artifacts at the time of freezing.

**Decision:** Accept the PaySim Silver artifacts as the current validated baseline for subsequent FINRISK NEXUS development.

**Change-control rule:** If any frozen artifact is regenerated or modified, recompute its checksum, rerun the applicable validation checks, and document the reason and updated validation status before accepting the changed artifact.


## 9. Freeze procedure

Run the following from the FINRISK NEXUS repository root in PowerShell.
This verifies that the expected artifacts exist and calculates their
current SHA-256 hashes.

``` powershell
$artifacts = @(
    ".\data\silver\paysim\transactions.parquet",
    ".\data\silver\paysim\transactions.metadata.json",
    ".\data\silver\paysim\data_quality_report.json"
)

foreach ($artifact in $artifacts) {
    if (-not (Test-Path $artifact -PathType Leaf)) {
        throw "Required freeze artifact is missing: $artifact"
    }

    Get-FileHash $artifact -Algorithm SHA256 |
        Select-Object Path, Algorithm, Hash
}
```

Compare the Parquet hash with the value in the Silver metadata. If it
differs, investigate before signing off; the file may have changed or
the recorded metadata may be stale.

After verifying the artifacts, preserve the hash output in the project
documentation or version-control history. Do not overwrite the validated
Parquet file or its evidence files silently. If the transformation
changes, create a new validation record and update the transformation
version according to the project's versioning policy.

### Optional: record the freeze hashes in a text file

After reviewing the hash output, run this command to save a fresh local
manifest:

``` powershell
$manifest = ".\data\silver\paysim\freeze_sha256.txt"

@(
    "# FINRISK NEXUS PaySim Silver freeze hashes"
    "# Generated from the current local artifacts; verify before sign-off."
    (Get-FileHash ".\data\silver\paysim\transactions.parquet" -Algorithm SHA256 |
        ForEach-Object { "$($_.Hash)  data/silver/paysim/transactions.parquet" })
    (Get-FileHash ".\data\silver\paysim\transactions.metadata.json" -Algorithm SHA256 |
        ForEach-Object { "$($_.Hash)  data/silver/paysim/transactions.metadata.json" })
    (Get-FileHash ".\data\silver\paysim\data_quality_report.json" -Algorithm SHA256 |
        ForEach-Object { "$($_.Hash)  data/silver/paysim/data_quality_report.json" })
) | Set-Content -Encoding utf8 $manifest

Get-Content $manifest
```

This optional command creates a local manifest; it does not make the
files immutable. Review the hashes and keep the manifest with the
validation record.

## 10. Freeze decision and remaining scope

**Current decision:** the observed structural and recorded data-quality
checks pass. The dataset is suitable to proceed to controlled
exploratory analysis, subject to the remaining semantic checks below.

Before modelling, still investigate:

1.  Fraud prevalence by transaction type.
2.  The meaning and temporal ordering of `step`.
3.  Whether features derived from balances or `isFlaggedFraud` could
    leak information unavailable at prediction time.
4.  The split strategy needed to prevent temporal or entity leakage,
    where relevant.
5.  Whether the source and transformed data satisfy the intended
    business rules beyond the checks already recorded.

The validation freeze does not certify model readiness, production
readiness, or absence of target leakage.

------------------------------------------------------------------------

**Prepared for:** FINRISK NEXUS project documentation\
**Evidence basis:** user-provided pipeline output, JSON report
screenshots, metadata screenshot, and independent Polars schema /
fraud-count output\
**Final hash verification:** pending local execution of Section 8
