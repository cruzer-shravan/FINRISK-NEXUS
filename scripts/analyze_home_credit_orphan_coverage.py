import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE = Path("data/raw/home_credit")
OUTPUT = Path("reports/data_exploration/home_credit")
OUTPUT.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000

CHECKS = [
    {
        "child": "bureau_balance.csv",
        "child_key": "SK_ID_BUREAU",
        "parent": "bureau.csv",
        "parent_key": "SK_ID_BUREAU",
    },
    {
        "child": "POS_CASH_balance.csv",
        "child_key": "SK_ID_PREV",
        "parent": "previous_application.csv",
        "parent_key": "SK_ID_PREV",
    },
    {
        "child": "credit_card_balance.csv",
        "child_key": "SK_ID_PREV",
        "parent": "previous_application.csv",
        "parent_key": "SK_ID_PREV",
    },
    {
        "child": "installments_payments.csv",
        "child_key": "SK_ID_PREV",
        "parent": "previous_application.csv",
        "parent_key": "SK_ID_PREV",
    },
]


def load_ids(filename, column):
    values = set()

    for chunk in pd.read_csv(
        BASE / filename,
        usecols=[column],
        dtype={column: "string"},
        chunksize=CHUNK_SIZE,
    ):
        values.update(chunk[column].dropna().tolist())

    return values


def main():
    print("Loading application customer identifiers...")

    application_ids = load_ids("application_train.csv", "SK_ID_CURR") | load_ids(
        "application_test.csv", "SK_ID_CURR"
    )

    results = []

    for check in CHECKS:
        child = check["child"]
        child_key = check["child_key"]
        parent = check["parent"]
        parent_key = check["parent_key"]

        print(f"\nAnalyzing {child}...")

        parent_ids = load_ids(parent, parent_key)

        orphan_counts = Counter()
        orphan_customers = defaultdict(set)

        total_rows = 0
        orphan_rows = 0
        customer_rows = 0

        has_customer_id = child != "bureau_balance.csv"

        usecols = [child_key]
        if has_customer_id:
            usecols.append("SK_ID_CURR")

        dtypes = {column: "string" for column in usecols}

        for chunk in pd.read_csv(
            BASE / child,
            usecols=usecols,
            dtype=dtypes,
            chunksize=CHUNK_SIZE,
        ):
            total_rows += len(chunk)

            orphan_mask = chunk[child_key].notna() & ~chunk[child_key].isin(parent_ids)

            orphan_chunk = chunk.loc[orphan_mask]

            orphan_rows += len(orphan_chunk)

            orphan_counts.update(orphan_chunk[child_key].dropna().tolist())

            if has_customer_id:
                for prev_id, group in orphan_chunk.groupby(child_key):
                    orphan_customers[prev_id].update(group["SK_ID_CURR"].dropna().tolist())

                customer_rows += int(orphan_chunk["SK_ID_CURR"].notna().sum())

        orphan_ids = set(orphan_counts)

        result = {
            "child_table": child,
            "parent_table": parent,
            "child_key": child_key,
            "parent_key": parent_key,
            "total_child_rows": total_rows,
            "orphan_rows": orphan_rows,
            "distinct_orphan_ids": len(orphan_ids),
            "orphan_rows_percentage": round(orphan_rows / total_rows * 100, 4) if total_rows else 0,
            "orphan_customer_analysis": None,
            "top_20_orphan_ids_by_row_count": [
                {
                    "id": identifier,
                    "rows": count,
                }
                for identifier, count in orphan_counts.most_common(20)
            ],
        }

        if has_customer_id:
            orphan_customer_ids = set()

            for customer_ids in orphan_customers.values():
                orphan_customer_ids.update(customer_ids)

            matched_customers = orphan_customer_ids & application_ids
            unmatched_customers = orphan_customer_ids - application_ids

            result["orphan_customer_analysis"] = {
                "customer_id_source": "SK_ID_CURR in child table",
                "distinct_customer_ids": len(orphan_customer_ids),
                "customers_found_in_applications": len(matched_customers),
                "customers_missing_from_applications": len(unmatched_customers),
                "customer_match_percentage": round(
                    len(matched_customers) / len(orphan_customer_ids) * 100, 4
                )
                if orphan_customer_ids
                else 0,
                "customer_analysis_limitation": (
                    "Customer membership only; does not establish "
                    "the missing previous application relationship."
                ),
            }
        else:
            result["orphan_customer_analysis"] = {
                "status": "not_available",
                "reason": (
                    "bureau_balance.csv does not contain SK_ID_CURR. "
                    "Its customer relationship requires bureau.csv."
                ),
            }

        results.append(result)

        print(
            f"Total rows: {total_rows:,} | "
            f"Orphan rows: {orphan_rows:,} | "
            f"Distinct orphan IDs: {len(orphan_ids):,}"
        )

        if has_customer_id:
            analysis = result["orphan_customer_analysis"]
            print(
                f"Customers found in applications: {analysis['customers_found_in_applications']:,}"
            )
            print(
                f"Customers missing from applications: "
                f"{analysis['customers_missing_from_applications']:,}"
            )

    output_path = OUTPUT / "home_credit_orphan_coverage_report.json"

    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nOrphan coverage analysis completed.")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
