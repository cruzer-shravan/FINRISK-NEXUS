from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = "data/silver/paysim/transactions.parquet"
DEFAULT_FEATURES = "data/gold/paysim/transaction_features.parquet"
DEFAULT_ANALYTICS = "data/gold/paysim/fraud_analytics.parquet"
DEFAULT_REPORT = "reports/data_quality/paysim_gold_validation.json"

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

FEATURE_COLUMNS = [
    *EXPECTED_COLUMNS,
    "origin_balance_error",
    "destination_balance_error",
    "log_amount",
    "origin_balance_mismatch",
    "destination_balance_mismatch",
]

MISMATCH_TOLERANCE = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def ensure_inside_project(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"{label} must be inside the project: {resolved}")
    return resolved


def run_pipeline(
    source: Path,
    features_output: Path,
    analytics_output: Path,
    report_output: Path,
) -> dict:
    source = ensure_inside_project(source, "Source")
    features_output = ensure_inside_project(features_output, "Features output")
    analytics_output = ensure_inside_project(analytics_output, "Analytics output")
    report_output = ensure_inside_project(report_output, "Report output")

    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".parquet":
        raise ValueError("Gold source must be a Parquet file.")

    outputs = [features_output, analytics_output, report_output]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing Gold artifacts. Inspect or archive them first:\n"
            + "\n".join(existing)
        )

    # Ensure all temporary/final artifacts are on their intended filesystems.
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    lazy = pl.scan_parquet(source)
    source_schema = lazy.collect_schema()
    source_columns = source_schema.names()
    if source_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected Silver schema or column order.\n"
            f"Expected: {EXPECTED_COLUMNS}\nReceived: {source_columns}"
        )

    source_rows = int(lazy.select(pl.len()).collect().item())
    source_sha256 = sha256_file(source)

    # Check source invariants required for safe feature calculations.
    source_checks = (
        lazy.select(
            pl.col("amount").is_null().sum().alias("null_amount"),
            (pl.col("amount") < 0).sum().alias("negative_amount"),
            pl.col("isFraud").is_null().sum().alias("null_isFraud"),
            (~pl.col("isFraud").is_in([0, 1])).sum().alias("invalid_isFraud"),
            pl.col("isFlaggedFraud").is_null().sum().alias("null_isFlaggedFraud"),
            (~pl.col("isFlaggedFraud").is_in([0, 1])).sum().alias("invalid_isFlaggedFraud"),
        )
        .collect()
        .row(0, named=True)
    )

    failed_checks = {k: int(v) for k, v in source_checks.items() if int(v) != 0}
    if failed_checks:
        raise ValueError(
            "Silver invariants failed; no Gold outputs were published:\n"
            + json.dumps(failed_checks, indent=2)
        )

    enriched = lazy.with_columns(
        (pl.col("oldbalanceOrg") - pl.col("newbalanceOrig") - pl.col("amount")).alias(
            "origin_balance_error"
        ),
        (pl.col("newbalanceDest") - pl.col("oldbalanceDest") - pl.col("amount")).alias(
            "destination_balance_error"
        ),
        pl.col("amount").log1p().alias("log_amount"),
    ).with_columns(
        (pl.col("origin_balance_error").abs() > MISMATCH_TOLERANCE)
        .cast(pl.Int8)
        .alias("origin_balance_mismatch"),
        (pl.col("destination_balance_error").abs() > MISMATCH_TOLERANCE)
        .cast(pl.Int8)
        .alias("destination_balance_mismatch"),
    )

    # isFraud is retained as the target; isFlaggedFraud is retained for auditing.
    # Neither is included in this file's declared MODEL_FEATURE_COLUMNS list below.
    MODEL_FEATURE_COLUMNS = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "origin_balance_error",
        "destination_balance_error",
        "origin_balance_mismatch",
        "destination_balance_mismatch",
        "log_amount",
    ]

    feature_tmp = features_output.with_name(features_output.stem + ".tmp.parquet")
    analytics_tmp = analytics_output.with_name(analytics_output.stem + ".tmp.parquet")

    try:
        enriched.sink_parquet(
            feature_tmp,
            compression="zstd",
            maintain_order=True,
        )

        # One compact analytics table with an explicit summary_level and group_value.
        by_type = (
            lazy.group_by("type")
            .agg(
                pl.len().alias("total_transactions"),
                pl.col("isFraud").sum().alias("fraud_transactions"),
                pl.col("isFlaggedFraud").sum().alias("flagged_transactions"),
            )
            .with_columns(
                pl.lit("transaction_type").alias("summary_level"),
                pl.col("type").alias("group_value"),
            )
            .select(
                "summary_level",
                "group_value",
                "total_transactions",
                "fraud_transactions",
                "flagged_transactions",
            )
        )

        by_step = (
            lazy.group_by("step")
            .agg(
                pl.len().alias("total_transactions"),
                pl.col("isFraud").sum().alias("fraud_transactions"),
                pl.col("isFlaggedFraud").sum().alias("flagged_transactions"),
            )
            .with_columns(
                pl.lit("simulation_step").alias("summary_level"),
                pl.col("step").cast(pl.String).alias("group_value"),
            )
            .select(
                "summary_level",
                "group_value",
                "total_transactions",
                "fraud_transactions",
                "flagged_transactions",
            )
        )

        pl.concat([by_type, by_step], how="vertical").sink_parquet(
            analytics_tmp,
            compression="zstd",
            maintain_order=True,
        )

        feature_check = pl.scan_parquet(feature_tmp)
        analytics_check = pl.scan_parquet(analytics_tmp)
        output_rows = int(feature_check.select(pl.len()).collect().item())
        output_columns = feature_check.collect_schema().names()

        if output_rows != source_rows:
            raise RuntimeError(
                f"Gold feature row count mismatch: source={source_rows}, output={output_rows}"
            )
        if output_columns != FEATURE_COLUMNS:
            raise RuntimeError(
                "Unexpected Gold feature schema/order.\n"
                f"Expected: {FEATURE_COLUMNS}\nReceived: {output_columns}"
            )

        # Verify label counts survived exactly in transaction-level output.
        label_counts_source = lazy.group_by("isFraud").len().sort("isFraud").collect()
        label_counts_output = feature_check.group_by("isFraud").len().sort("isFraud").collect()
        if not label_counts_source.equals(label_counts_output):
            raise RuntimeError("Fraud-label counts changed in Gold feature output.")

        # Verify aggregate transaction counts reconcile to the Silver row count.
        analytics = analytics_check.collect()
        type_total = int(
            analytics.filter(pl.col("summary_level") == "transaction_type")
            .get_column("total_transactions")
            .sum()
        )
        if type_total != source_rows:
            raise RuntimeError(
                f"Transaction-type analytics do not reconcile: {type_total} != {source_rows}"
            )

        # Publish only after both Parquet files pass validation.
        feature_tmp.replace(features_output)
        analytics_tmp.replace(analytics_output)

    except Exception:
        feature_tmp.unlink(missing_ok=True)
        analytics_tmp.unlink(missing_ok=True)
        features_output.unlink(missing_ok=True)
        analytics_output.unlink(missing_ok=True)
        raise

    fraud_distribution = (
        pl.scan_parquet(features_output).group_by("isFraud").len().sort("isFraud").collect()
    )
    fraud_counts = {str(row["isFraud"]): int(row["len"]) for row in fraud_distribution.to_dicts()}

    quality_report = {
        "dataset": "PaySim Gold layer",
        "source_file": relative_path(source),
        "transaction_features_file": relative_path(features_output),
        "fraud_analytics_file": relative_path(analytics_output),
        "source_rows": source_rows,
        "gold_transaction_rows": output_rows,
        "source_columns": len(source_columns),
        "gold_transaction_columns": len(output_columns),
        "fraud_label_counts": fraud_counts,
        "source_invariant_checks": {k: int(v) for k, v in source_checks.items()},
        "balance_mismatch_tolerance": MISMATCH_TOLERANCE,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "target_column": "isFraud",
        "audit_only_column": "isFlaggedFraud",
        "checks": {
            "row_count_preserved": output_rows == source_rows,
            "feature_schema_matches": output_columns == FEATURE_COLUMNS,
            "fraud_label_counts_preserved": True,
            "transaction_type_analytics_reconciles": type_total == source_rows,
        },
        "validation_status": "passed",
        "source_sha256": source_sha256,
        "transaction_features_sha256": sha256_file(features_output),
        "fraud_analytics_sha256": sha256_file(analytics_output),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    try:
        report_output.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
    except Exception:
        features_output.unlink(missing_ok=True)
        analytics_output.unlink(missing_ok=True)
        report_output.unlink(missing_ok=True)
        raise

    print("\nPAYSIM GOLD PIPELINE COMPLETED")
    print(f"Source rows: {source_rows:,}")
    print(f"Gold transaction rows: {output_rows:,}")
    print(f"Gold transaction columns: {len(output_columns)}")
    print(f"Transaction features: {features_output}")
    print(f"Fraud analytics: {analytics_output}")
    print(f"Validation report: {report_output}")
    return quality_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and validate the PaySim Gold layer from frozen Silver Parquet."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--features-output", default=DEFAULT_FEATURES)
    parser.add_argument("--analytics-output", default=DEFAULT_ANALYTICS)
    parser.add_argument("--report-output", default=DEFAULT_REPORT)
    args = parser.parse_args()

    run_pipeline(
        PROJECT_ROOT / args.source,
        PROJECT_ROOT / args.features_output,
        PROJECT_ROOT / args.analytics_output,
        PROJECT_ROOT / args.report_output,
    )


if __name__ == "__main__":
    main()
