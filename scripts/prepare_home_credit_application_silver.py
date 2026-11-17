import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SOURCE = Path("data/bronze/home_credit/application_train.parquet")
OUTPUT_DIR = Path("data/silver/home_credit")
OUTPUT = OUTPUT_DIR / "application_train.parquet"
METADATA = OUTPUT_DIR / "application_train.metadata.json"

IDENTIFIER_COLUMNS = ["SK_ID_CURR"]
TARGET_COLUMN = "TARGET"

CATEGORICAL_COLUMNS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading Bronze application dataset...")
    df = pd.read_parquet(SOURCE)

    original_rows, original_columns = df.shape

    required_columns = IDENTIFIER_COLUMNS + [TARGET_COLUMN] + CATEGORICAL_COLUMNS

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

    numeric_columns = [
        col
        for col in df.columns
        if col not in (IDENTIFIER_COLUMNS + [TARGET_COLUMN] + CATEGORICAL_COLUMNS)
    ]

    conversion_summary = {
        "integer_columns": [],
        "float_columns": [],
        "categorical_columns": CATEGORICAL_COLUMNS,
    }

    print("Converting identifier...")
    for col in IDENTIFIER_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")

        failures = (df[col].notna() & values.isna()).sum()
        if failures:
            raise ValueError(f"{col}: {failures} conversion failures")

        df[col] = values.astype("Int64")

    print("Converting target...")
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    target_failures = (df[TARGET_COLUMN].notna() & target.isna()).sum()

    if target_failures:
        raise ValueError(f"TARGET conversion failures: {target_failures}")

    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("TARGET contains missing or unexpected values")

    df[TARGET_COLUMN] = target.astype("int8")

    print("Converting numeric features...")
    for col in numeric_columns:
        original = df[col]
        converted = pd.to_numeric(original, errors="coerce")

        failures = (original.notna() & converted.isna()).sum()

        if failures:
            raise ValueError(f"{col}: {failures} numeric conversion failures")

        non_missing = converted.dropna()

        if len(non_missing) == 0 or non_missing.mod(1).eq(0).all():
            df[col] = converted.astype("Int64")
            conversion_summary["integer_columns"].append(col)
        else:
            df[col] = converted.astype("Float64")
            conversion_summary["float_columns"].append(col)

    print("Preserving categorical features...")
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("string")

    # Final structural checks
    if df.shape != (original_rows, original_columns):
        raise ValueError("Row or column count changed unexpectedly")

    if df["SK_ID_CURR"].duplicated().any():
        raise ValueError("Duplicate customer identifiers detected")

    if df["TARGET"].isna().any():
        raise ValueError("Missing target values detected")

    if df["SK_ID_CURR"].isna().any():
        raise ValueError("Missing customer identifiers detected")

    print("Writing Silver Parquet...")
    df.to_parquet(
        OUTPUT,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    metadata = {
        "dataset_layer": "silver",
        "dataset_name": "home_credit_application_train",
        "source_file": str(SOURCE),
        "output_file": str(OUTPUT),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "format": "parquet",
        "compression": "zstd",
        "rows": len(df),
        "columns": len(df.columns),
        "source_sha256": sha256_file(SOURCE),
        "output_sha256": sha256_file(OUTPUT),
        "identifier_columns": IDENTIFIER_COLUMNS,
        "target_column": TARGET_COLUMN,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "integer_feature_columns": conversion_summary["integer_columns"],
        "float_feature_columns": conversion_summary["float_columns"],
        "target_distribution": {
            str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().items()
        },
        "missing_values_by_column": {
            col: int(count) for col, count in df.isna().sum().items() if count > 0
        },
        "dtype_summary": {
            str(dtype): int(count) for dtype, count in df.dtypes.astype(str).value_counts().items()
        },
        "validation_status": "passed",
    }

    METADATA.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nSilver transformation completed.")
    print("Output:", OUTPUT)
    print("Metadata:", METADATA)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Integer features:", len(conversion_summary["integer_columns"]))
    print("Float features:", len(conversion_summary["float_columns"]))
    print("Categorical features:", len(CATEGORICAL_COLUMNS))
    print("Target distribution:")
    print(df[TARGET_COLUMN].value_counts())


if __name__ == "__main__":
    main()
