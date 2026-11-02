from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = (PROJECT_ROOT / args.source).resolve()
    output = (PROJECT_ROOT / args.output).resolve()

    if not source.is_relative_to(PROJECT_ROOT):
        raise ValueError("Source must be inside the project.")
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Output must be inside the project.")
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".csv":
        raise ValueError("Source must be a CSV file.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)

    # Only the reference file has been confirmed as cp1252.
    if source.name == "HomeCredit_columns_description.csv":
        text = source.read_bytes().decode("cp1252")
        csv_source = io.BytesIO(text.encode("utf-8"))
        frame = pl.read_csv(
            csv_source,
            infer_schema=False,
            try_parse_dates=False,
        )
        source_encoding = "cp1252"
        frame.write_parquet(output, compression="zstd")
        source_rows = frame.height
        source_columns = frame.columns
    else:
        frame = pl.scan_csv(
            source,
            infer_schema=False,
            try_parse_dates=False,
            encoding="utf8",
        )
        source_columns = frame.collect_schema().names()
        source_rows = frame.select(pl.len().alias("rows")).collect().item()
        source_encoding = "utf8"
        frame.sink_parquet(
            output,
            compression="zstd",
            maintain_order=True,
        )

    try:
        check = pl.scan_parquet(output)
        output_columns = check.collect_schema().names()
        output_rows = check.select(pl.len().alias("rows")).collect().item()

        if source_rows != output_rows:
            raise RuntimeError(f"Row mismatch: {source_rows} != {output_rows}")
        if source_columns != output_columns:
            raise RuntimeError("Column names/order mismatch.")

        if not all(dtype == pl.String for dtype in check.collect_schema().dtypes()):
            raise RuntimeError("Unexpected non-string Bronze column.")

        report = {
            "dataset_layer": "bronze",
            "source_file": source.relative_to(PROJECT_ROOT).as_posix(),
            "output_file": output.relative_to(PROJECT_ROOT).as_posix(),
            "source_encoding": source_encoding,
            "format": "parquet",
            "compression": "zstd",
            "source_rows": source_rows,
            "parquet_rows": output_rows,
            "source_columns": len(source_columns),
            "parquet_columns": len(output_columns),
            "column_names": output_columns,
            "column_names_match": True,
            "row_counts_match": True,
            "source_size_bytes": source.stat().st_size,
            "parquet_size_bytes": output.stat().st_size,
            "source_sha256": sha256_file(source),
            "parquet_sha256": sha256_file(output),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "validation_status": "passed",
        }

        metadata = output.with_suffix(".metadata.json")
        metadata.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print("\nCONVERSION VALIDATION PASSED")
        print(f"Rows: {output_rows:,}")
        print(f"Columns: {len(output_columns)}")
        print(f"Parquet size: {output.stat().st_size / 1_000_000:.2f} MB")
        print(f"Metadata: {metadata}")

    except Exception:
        output.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
