from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE = "data/bronze/paysim/PS_20174392719_1491204439457_log.parquet"
DEFAULT_OUTPUT = "data/silver/paysim/transactions.parquet"

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

NUMERIC_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

BALANCE_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def run_pipeline(source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()

    if not source.is_relative_to(PROJECT_ROOT):
        raise ValueError("Source must be inside the project.")

    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Output must be inside the project.")

    if not source.is_file():
        raise FileNotFoundError(source)

    if source.suffix.lower() != ".parquet":
        raise ValueError("Silver source must be a Parquet file.")

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    metadata_path = output.with_suffix(".metadata.json")
    quality_path = output.with_name("data_quality_report.json")

    if metadata_path.exists() or quality_path.exists():
        raise FileExistsError(
            "Silver metadata or quality report already exists. "
            "Inspect existing artifacts before proceeding."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    # Inspect the Bronze schema before processing.
    lazy = pl.scan_parquet(source)
    schema = lazy.collect_schema()
    source_columns = schema.names()

    if source_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected Bronze schema or column order.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Received: {source_columns}"
        )

    source_rows = lazy.select(pl.len()).collect().item()
    source_sha256 = sha256_file(source)

    # Explicitly cast the columns; do not infer their types.
    typed = lazy.with_columns(
        pl.col("step").cast(pl.Int64, strict=False),
        *[pl.col(column).cast(pl.Float64, strict=False) for column in NUMERIC_COLUMNS],
        pl.col("isFraud").cast(pl.Int8, strict=False),
        pl.col("isFlaggedFraud").cast(pl.Int8, strict=False),
    )

    # Detect invalid values before creating the final output.
    invalid_expressions = [
        pl.col("step").is_null().sum().alias("invalid_step"),
        pl.col("type").is_null().sum().alias("null_type"),
        pl.col("nameOrig").is_null().sum().alias("null_origin"),
        pl.col("nameDest").is_null().sum().alias("null_destination"),
        *[pl.col(column).is_null().sum().alias(f"invalid_{column}") for column in NUMERIC_COLUMNS],
        pl.col("isFraud").is_null().sum().alias("invalid_isFraud"),
        pl.col("isFlaggedFraud").is_null().sum().alias("invalid_isFlaggedFraud"),
        (pl.col("step") < 0).sum().alias("negative_step"),
        (pl.col("amount") < 0).sum().alias("negative_amount"),
        *[(pl.col(column) < 0).sum().alias(f"negative_{column}") for column in BALANCE_COLUMNS],
        (~pl.col("type").is_in(TRANSACTION_TYPES)).sum().alias("unexpected_transaction_type"),
        (~pl.col("isFraud").is_in([0, 1])).sum().alias("invalid_fraud_flag"),
        (~pl.col("isFlaggedFraud").is_in([0, 1])).sum().alias("invalid_flagged_fraud"),
    ]

    quality = typed.select(invalid_expressions).collect().row(0, named=True)

    # Invalid values must not silently enter the trusted Silver dataset.
    failures = {name: int(count) for name, count in quality.items() if count != 0}

    if failures:
        raise ValueError(
            "Silver quality checks failed. No output was written.\n"
            + json.dumps(failures, indent=2)
        )

    silver = typed.with_columns(
        pl.col("type").cast(pl.String),
        pl.col("nameOrig").cast(pl.String),
        pl.col("nameDest").cast(pl.String),
    )

    # Stage the result so a failed write does not leave a final artifact.
    temporary = output.with_name(output.stem + ".tmp.parquet")

    try:
        silver.sink_parquet(
            temporary,
            compression="zstd",
            maintain_order=True,
        )

        check = pl.scan_parquet(temporary)
        output_schema = check.collect_schema()

        if check.select(pl.len()).collect().item() != source_rows:
            raise RuntimeError("Silver row count differs from Bronze.")

        if output_schema.names() != EXPECTED_COLUMNS:
            raise RuntimeError("Silver column names/order mismatch.")

        expected_types = {
            "step": pl.Int64,
            "type": pl.String,
            "amount": pl.Float64,
            "nameOrig": pl.String,
            "oldbalanceOrg": pl.Float64,
            "newbalanceOrig": pl.Float64,
            "nameDest": pl.String,
            "oldbalanceDest": pl.Float64,
            "newbalanceDest": pl.Float64,
            "isFraud": pl.Int8,
            "isFlaggedFraud": pl.Int8,
        }

        actual_types = dict(output_schema)

        if actual_types != expected_types:
            raise RuntimeError(f"Unexpected Silver data types: {actual_types}")

        output_rows = check.select(pl.len()).collect().item()
        output_columns = output_schema.names()

        # Verify that the fraud indicators remain valid after writing.
        persisted_quality = (
            check.select(
                (~pl.col("isFraud").is_in([0, 1])).sum().alias("invalid_fraud_flag"),
                (~pl.col("isFlaggedFraud").is_in([0, 1])).sum().alias("invalid_flagged_fraud"),
            )
            .collect()
            .row(0, named=True)
        )

        if any(persisted_quality.values()):
            raise RuntimeError("Persisted fraud flag validation failed.")

        # Atomic rename on the same filesystem.
        temporary.replace(output)

    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    quality_report = {
        "dataset": "PaySim transactions",
        "source_file": relative_path(source),
        "output_file": relative_path(output),
        "rows_checked": source_rows,
        "quality_checks": {key: int(value) for key, value in quality.items()},
        "validation_status": "passed",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    metadata = {
        "dataset_layer": "silver",
        "dataset": "PaySim transactions",
        "source_file": relative_path(source),
        "output_file": relative_path(output),
        "format": "parquet",
        "compression": "zstd",
        "transformation_version": "1.0.0",
        "source_sha256": source_sha256,
        "parquet_sha256": sha256_file(output),
        "source_rows": source_rows,
        "silver_rows": output_rows,
        "source_columns": len(source_columns),
        "silver_columns": len(output_columns),
        "column_names": output_columns,
        "schema": {name: str(dtype) for name, dtype in output_schema.items()},
        "row_counts_match": source_rows == output_rows,
        "source_size_bytes": source.stat().st_size,
        "parquet_size_bytes": output.stat().st_size,
        "quality_report": relative_path(quality_path),
        "validation_status": "passed",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    try:
        quality_path.write_text(
            json.dumps(quality_report, indent=2),
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
    except Exception:
        output.unlink(missing_ok=True)
        quality_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise

    print("\nPAYSIM SILVER PIPELINE COMPLETED")
    print(f"Rows: {output_rows:,}")
    print(f"Columns: {len(output_columns)}")
    print(f"Parquet: {output}")
    print(f"Metadata: {metadata_path}")
    print(f"Quality report: {quality_path}")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and transform Bronze PaySim data to Silver."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = (PROJECT_ROOT / args.source).resolve()
    output = (PROJECT_ROOT / args.output).resolve()

    run_pipeline(source, output)


if __name__ == "__main__":
    main()
