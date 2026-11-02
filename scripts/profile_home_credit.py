import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path("data/raw/home_credit")
OUTPUT_DIR = Path("reports/data_exploration/home_credit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABLES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
]

CHUNK_SIZE = 100_000


def profile_csv(filename: str) -> dict:
    path = BASE_DIR / filename

    if not path.exists():
        return {
            "file": filename,
            "status": "missing",
            "message": f"File not found: {path.as_posix()}",
        }

    total_rows = 0
    columns = None
    dtypes = {}
    missing_counts = {}
    exact_duplicate_count = 0
    target_counts = {}
    numeric_min = {}
    numeric_max = {}
    numeric_negative_counts = {}
    key_counts = {}
    key_missing_counts = {}

    # Keys used for preliminary validation only.
    candidate_keys = {
        "application_train.csv": ["SK_ID_CURR"],
        "application_test.csv": ["SK_ID_CURR"],
        "bureau.csv": ["SK_ID_BUREAU"],
        "bureau_balance.csv": ["SK_ID_BUREAU", "MONTHS_BALANCE"],
        "previous_application.csv": ["SK_ID_PREV"],
        "POS_CASH_balance.csv": ["SK_ID_PREV", "MONTHS_BALANCE"],
        "credit_card_balance.csv": ["SK_ID_PREV", "MONTHS_BALANCE"],
        "installments_payments.csv": [
            "SK_ID_PREV",
            "NUM_INSTALMENT_VERSION",
            "NUM_INSTALMENT_NUMBER",
        ],
    }.get(filename, [])

    # For full-file duplicate checks, use pandas duplicated
    # within each chunk only. Cross-chunk duplicates are not
    # counted by this first-pass implementation.
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)

        if columns is None:
            columns = list(chunk.columns)

        for col in chunk.columns:
            dtypes.setdefault(col, str(chunk[col].dtype))
            missing_counts[col] = missing_counts.get(col, 0) + int(chunk[col].isna().sum())

            if pd.api.types.is_numeric_dtype(chunk[col]):
                if chunk[col].notna().any():
                    chunk_min = chunk[col].min()
                    chunk_max = chunk[col].max()

                    numeric_min[col] = min(
                        numeric_min.get(col, chunk_min),
                        chunk_min,
                    )
                    numeric_max[col] = max(
                        numeric_max.get(col, chunk_max),
                        chunk_max,
                    )

                numeric_negative_counts[col] = numeric_negative_counts.get(col, 0) + int(
                    (chunk[col] < 0).sum()
                )

        exact_duplicate_count += int(chunk.duplicated().sum())

        if "TARGET" in chunk.columns:
            counts = chunk["TARGET"].value_counts(dropna=False)
            for label, count in counts.items():
                label = str(label)
                target_counts[label] = target_counts.get(label, 0) + int(count)

        for key in candidate_keys:
            if key in chunk.columns:
                key_missing_counts[key] = key_missing_counts.get(key, 0) + int(
                    chunk[key].isna().sum()
                )
                key_counts.setdefault(key, set()).update(chunk[key].dropna().unique().tolist())

    report = {
        "file": filename,
        "status": "profiled",
        "rows": total_rows,
        "column_count": len(columns or []),
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": missing_counts,
        "missing_percent": {
            col: round(count / total_rows * 100, 4) if total_rows else None
            for col, count in missing_counts.items()
        },
        "exact_duplicate_rows_within_chunks": exact_duplicate_count,
        "target_counts": target_counts,
        "numeric_min": numeric_min,
        "numeric_max": numeric_max,
        "numeric_negative_counts": numeric_negative_counts,
        "candidate_key_distinct_counts": {key: len(values) for key, values in key_counts.items()},
        "candidate_key_missing_counts": key_missing_counts,
        "limitations": [
            "Duplicate rows spanning different chunks are not detected.",
            "Distinct key counts are accumulated in memory.",
            "Candidate keys require business-semantic validation.",
            "Referential integrity is not checked by this script.",
        ],
    }

    return report


def main():
    results = []

    for filename in TABLES:
        print(f"Profiling: {filename}")
        result = profile_csv(filename)
        results.append(result)

        print(
            f"Status: {result['status']}"
            + (f" | Rows: {result['rows']:,}" if result["status"] == "profiled" else "")
        )

    output_path = OUTPUT_DIR / "home_credit_profile_summary.json"
    output_path.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
