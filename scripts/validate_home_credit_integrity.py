import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path("data/raw/home_credit")
OUTPUT_DIR = Path("reports/data_exploration/home_credit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000

# Read only the columns needed for identifier validation.
TABLES = {
    "application_train.csv": ["SK_ID_CURR"],
    "application_test.csv": ["SK_ID_CURR"],
    "bureau.csv": ["SK_ID_CURR", "SK_ID_BUREAU"],
    "bureau_balance.csv": ["SK_ID_BUREAU"],
    "previous_application.csv": ["SK_ID_CURR", "SK_ID_PREV"],
    "POS_CASH_balance.csv": ["SK_ID_PREV"],
    "credit_card_balance.csv": ["SK_ID_PREV"],
    "installments_payments.csv": ["SK_ID_PREV"],
}


def read_identifier_values(filename, column):
    """Collect non-null identifiers from a CSV column."""
    path = BASE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    values = set()
    row_count = 0
    missing_count = 0

    for chunk in pd.read_csv(
        path,
        usecols=[column],
        chunksize=CHUNK_SIZE,
    ):
        series = chunk[column]
        row_count += len(series)
        missing_count += int(series.isna().sum())
        values.update(series.dropna().tolist())

    return {
        "values": values,
        "rows": row_count,
        "missing": missing_count,
    }


def main():
    results = {}

    # Load the identifier columns needed for relationship checks.
    identifiers = {}

    for filename, columns in TABLES.items():
        print(f"Reading identifiers: {filename}")
        identifiers[filename] = {}

        for column in columns:
            identifiers[filename][column] = read_identifier_values(filename, column)

    def values(filename, column):
        return identifiers[filename][column]["values"]

    def summary(filename, column):
        item = identifiers[filename][column]
        distinct_count = len(item["values"])

        return {
            "rows": item["rows"],
            "missing_count": item["missing"],
            "distinct_non_null_count": distinct_count,
            "duplicate_non_null_count": (item["rows"] - item["missing"] - distinct_count),
        }

    # Summarize identifiers. Duplicate counts are exact for
    # individual identifier columns across the complete file.
    for filename, columns in TABLES.items():
        results[filename] = {column: summary(filename, column) for column in columns}

    train_ids = values("application_train.csv", "SK_ID_CURR")
    test_ids = values("application_test.csv", "SK_ID_CURR")
    all_application_ids = train_ids | test_ids

    results["train_test_overlap"] = {
        "overlapping_customer_ids": len(train_ids & test_ids),
        "status": ("PASS" if not (train_ids & test_ids) else "REVIEW"),
    }

    # Validate foreign-key membership. These checks establish
    # whether child IDs exist in the referenced parent table.
    relationships = [
        (
            "bureau.csv",
            "SK_ID_CURR",
            "application_train.csv",
            "SK_ID_CURR",
            "Bureau customer IDs -> all application IDs",
        ),
        (
            "previous_application.csv",
            "SK_ID_CURR",
            "application_train.csv",
            "SK_ID_CURR",
            "Previous-application customer IDs -> all application IDs",
        ),
        (
            "bureau_balance.csv",
            "SK_ID_BUREAU",
            "bureau.csv",
            "SK_ID_BUREAU",
            "Bureau-balance IDs -> bureau records",
        ),
        (
            "POS_CASH_balance.csv",
            "SK_ID_PREV",
            "previous_application.csv",
            "SK_ID_PREV",
            "POS-CASH IDs -> previous applications",
        ),
        (
            "credit_card_balance.csv",
            "SK_ID_PREV",
            "previous_application.csv",
            "SK_ID_PREV",
            "Credit-card IDs -> previous applications",
        ),
        (
            "installments_payments.csv",
            "SK_ID_PREV",
            "previous_application.csv",
            "SK_ID_PREV",
            "Installment IDs -> previous applications",
        ),
    ]

    results["relationships"] = []

    for (
        child_file,
        child_column,
        parent_file,
        parent_column,
        description,
    ) in relationships:
        child = values(child_file, child_column)

        if description in {
            "Bureau customer IDs -> all application IDs",
            "Previous-application customer IDs -> all application IDs",
        }:
            parent = all_application_ids
        else:
            parent = values(parent_file, parent_column)

        unmatched = child - parent

        results["relationships"].append(
            {
                "relationship": description,
                "distinct_child_ids": len(child),
                "distinct_parent_ids": len(parent),
                "unmatched_distinct_child_ids": len(unmatched),
                "sample_unmatched_ids": sorted(unmatched, key=str)[:10],
                "status": "PASS" if not unmatched else "REVIEW",
            }
        )

    output_path = OUTPUT_DIR / "home_credit_integrity_report.json"
    output_path.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nIntegrity validation completed.")
    print(f"Report saved to: {output_path}")

    print("\nTrain/test overlap:")
    print(results["train_test_overlap"])

    print("\nRelationship results:")
    for item in results["relationships"]:
        print(
            f"{item['status']}: {item['relationship']} "
            f"| unmatched IDs: "
            f"{item['unmatched_distinct_child_ids']:,}"
        )


if __name__ == "__main__":
    main()
