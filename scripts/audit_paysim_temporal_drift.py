from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = ROOT / "data" / "gold" / "paysim" / "transaction_features.parquet"

REPORT_DIR = ROOT / "reports" / "paysim"
OUTPUT_DIR = REPORT_DIR / "temporal_drift"

TEMPORAL_COLUMN = "step"
TARGET = "isFraud"
TRANSACTION_TYPE = "type"

TRAIN_END_STEP = 323
VALIDATION_END_STEP = 378
INTERVAL_SIZE = 50


def markdown_table(df, percentage_columns=None):
    """Convert a DataFrame into a Markdown table."""

    percentage_columns = percentage_columns or []

    display_df = df.copy()

    for column in percentage_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: f"{value:.6%}")

    headers = list(display_df.columns)

    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in display_df.iterrows():
        values = []

        for value in row:
            values.append(str(value))

        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def main():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Gold dataset not found: {SOURCE_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PaySim Gold dataset...")

    df = pd.read_parquet(
        SOURCE_PATH,
        columns=[
            TEMPORAL_COLUMN,
            TRANSACTION_TYPE,
            TARGET,
        ],
    )

    print(f"Source transactions: {len(df):,}")

    # --------------------------------------------------
    # 1. Fraud rate by 50-step interval
    # --------------------------------------------------

    df["period"] = ((df[TEMPORAL_COLUMN] - 1) // INTERVAL_SIZE) * INTERVAL_SIZE + 1

    interval_summary = (
        df.groupby("period")
        .agg(
            transactions=(TARGET, "size"),
            fraud_cases=(TARGET, "sum"),
            fraud_rate=(TARGET, "mean"),
            unique_steps=(TEMPORAL_COLUMN, "nunique"),
        )
        .reset_index()
    )

    interval_path = OUTPUT_DIR / "fraud_by_50_step_interval.csv"

    interval_summary.to_csv(
        interval_path,
        index=False,
    )

    # --------------------------------------------------
    # 2. Fraud rate by transaction type
    # --------------------------------------------------

    type_summary = (
        df.groupby(TRANSACTION_TYPE)
        .agg(
            transactions=(TARGET, "size"),
            fraud_cases=(TARGET, "sum"),
            fraud_rate=(TARGET, "mean"),
        )
        .sort_values(
            "fraud_rate",
            ascending=False,
        )
        .reset_index()
    )

    type_path = OUTPUT_DIR / "fraud_by_transaction_type.csv"

    type_summary.to_csv(
        type_path,
        index=False,
    )

    # --------------------------------------------------
    # 3. Fraud rate by type across temporal splits
    # --------------------------------------------------

    df["period_group"] = pd.Series(
        "Test (379-743)",
        index=df.index,
        dtype="object",
    )

    df.loc[
        df[TEMPORAL_COLUMN] <= TRAIN_END_STEP,
        "period_group",
    ] = "Early (1-323)"

    df.loc[
        (df[TEMPORAL_COLUMN] > TRAIN_END_STEP) & (df[TEMPORAL_COLUMN] <= VALIDATION_END_STEP),
        "period_group",
    ] = "Validation (324-378)"

    period_order = [
        "Early (1-323)",
        "Validation (324-378)",
        "Test (379-743)",
    ]

    df["period_group"] = pd.Categorical(
        df["period_group"],
        categories=period_order,
        ordered=True,
    )

    comparison = (
        df.groupby(
            ["period_group", TRANSACTION_TYPE],
            observed=True,
        )
        .agg(
            transactions=(TARGET, "size"),
            fraud_cases=(TARGET, "sum"),
            fraud_rate=(TARGET, "mean"),
        )
        .reset_index()
    )

    comparison_path = OUTPUT_DIR / "fraud_by_period_and_type.csv"

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # --------------------------------------------------
    # 4. Overall split prevalence
    # --------------------------------------------------

    split_summary = (
        df.groupby("period_group", observed=True)
        .agg(
            transactions=(TARGET, "size"),
            fraud_cases=(TARGET, "sum"),
            fraud_rate=(TARGET, "mean"),
            min_step=(TEMPORAL_COLUMN, "min"),
            max_step=(TEMPORAL_COLUMN, "max"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # 5. Generate Markdown report
    # --------------------------------------------------

    report = f"""# FINRISK NEXUS — PaySim Temporal Drift Audit

## 1. Audit overview

**Dataset:** `data/gold/paysim/transaction_features.parquet`

**Target:** `{TARGET}`

**Temporal column:** `{TEMPORAL_COLUMN}`

**Interval size:** {INTERVAL_SIZE} simulation steps

**Total transactions analysed:** {len(df):,}

**Analysis type:** Descriptive temporal distribution analysis.

This audit examines transaction volume and observed fraud prevalence
across simulation periods and transaction types.

It does not perform statistical drift testing, model retraining,
threshold optimization, or feature selection.

## 2. Fraud prevalence across modeling periods

{markdown_table(split_summary, ["fraud_rate"])}

The test period has a different observed fraud prevalence from the
training and validation periods.

This difference should be considered when interpreting the locked
test evaluation.

## 3. Fraud rate by 50-step interval

{markdown_table(interval_summary, ["fraud_rate"])}

The results show that fraud prevalence varies across simulation
intervals rather than remaining constant throughout the dataset.

The intervals are descriptive groupings and should not be interpreted
as evidence of the cause of the observed changes.

## 4. Fraud rate by transaction type

{markdown_table(type_summary, ["fraud_rate"])}

The transaction-type analysis shows how fraud cases are distributed
across the available transaction categories.

A zero observed fraud rate does not establish that a transaction
category is inherently incapable of fraud.

## 5. Fraud rate by transaction type and modeling period

{markdown_table(comparison, ["fraud_rate"])}

This comparison highlights how the observed fraud prevalence for
different transaction types varies across the early, validation,
and test periods.

## 6. Key observations

1. Fraud prevalence varies across the simulation timeline.
2. The validation and test periods have substantially different
   observed fraud rates.
3. Fraud prevalence also differs across transaction types.
4. These observations indicate temporal distribution differences
   that warrant further investigation.
5. The underlying cause of these differences has not been established.

## 7. Methodology and limitations

- Source data was read from the existing PaySim Gold Parquet dataset.
- Only `step`, `type`, and `isFraud` were loaded.
- Temporal intervals were grouped into 50-step periods.
- Modeling-period boundaries were aligned with the existing
  chronological split manifest.
- Fraud prevalence was calculated as fraud cases divided by
  transaction count.
- This is descriptive analysis, not a formal statistical drift test.
- No model predictions or thresholds were changed.
- No training, validation, or test datasets were modified.
- No model was retrained during this audit.

The audit does not establish whether the observed differences are
caused by changes in transaction behavior, simulation design,
fraud-generation mechanisms, or other factors.

## 8. Output artifacts

| Artifact | Description |
|---|---|
| `temporal_drift/fraud_by_50_step_interval.csv` | Fraud statistics by 50-step interval |
| `temporal_drift/fraud_by_transaction_type.csv` | Fraud statistics by transaction type |
| `temporal_drift/fraud_by_period_and_type.csv` | Fraud statistics by period and transaction type |
| `temporal_drift_report.md` | Consolidated temporal drift report |

## 9. Evaluation integrity

This audit is supplementary descriptive analysis.

The existing Decision Tree and Random Forest locked-test evaluation
results remain frozen. The test partition must not be reused for
threshold tuning, feature selection, hyperparameter optimization,
or iterative model selection.

---

**Project:** FINRISK NEXUS

**Dataset:** PaySim
"""

    report_path = REPORT_DIR / "temporal_drift_report.md"

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 6. Display completion summary
    # --------------------------------------------------

    print("\n========== TEMPORAL DRIFT AUDIT ==========")

    print("\nFraud prevalence by modeling period:")
    print(split_summary.to_string(index=False))

    print("\nFraud rate by transaction type:")
    print(type_summary.to_string(index=False))

    print("\nFraud rate by 50-step interval:")
    print(interval_summary.to_string(index=False))

    print("\n========== OUTPUT FILES ==========")

    print(f"Markdown report: {report_path}")
    print(f"Interval CSV: {interval_path}")
    print(f"Transaction type CSV: {type_path}")
    print(f"Period/type CSV: {comparison_path}")

    print("\nTemporal drift audit completed successfully.")


if __name__ == "__main__":
    main()
