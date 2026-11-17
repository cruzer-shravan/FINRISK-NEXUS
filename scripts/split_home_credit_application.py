import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data/silver/home_credit/application_train.parquet"

OUTPUT_DIR = ROOT / "data/modeling/home_credit/application_only"

TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"
RANDOM_STATE = 42


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

    output_files = [
        OUTPUT_DIR / "train.parquet",
        OUTPUT_DIR / "validation.parquet",
        OUTPUT_DIR / "test.parquet",
    ]

    if any(path.exists() for path in output_files):
        raise FileExistsError(
            "One or more split files already exist. "
            "Inspect or archive them before creating new splits."
        )

    print("Loading Silver application dataset...")

    df = pd.read_parquet(SOURCE)

    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    if df[TARGET].isna().any():
        raise ValueError("Missing target values detected.")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Unexpected target values.")

    # First split: 70% training, 30% temporary.
    train, temporary = train_test_split(
        df,
        test_size=0.30,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )

    # Second split: 15% validation, 15% test.
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        stratify=temporary[TARGET],
        random_state=RANDOM_STATE,
    )

    partitions = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    # Validate partition integrity.
    id_sets = {name: set(partition[ID_COLUMN]) for name, partition in partitions.items()}

    assert id_sets["train"].isdisjoint(id_sets["validation"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["validation"].isdisjoint(id_sets["test"])

    assert sum(len(partition) for partition in partitions.values()) == len(df)

    split_metadata = {
        "dataset": "home_credit_application_only",
        "source": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "random_state": RANDOM_STATE,
        "strategy": "stratified_random_split",
        "target": TARGET,
        "identifier": ID_COLUMN,
        "proportions": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "partitions": {},
        "validation": {
            "customer_ids_disjoint": True,
            "total_rows_preserved": True,
            "target_stratification": True,
        },
    }

    for name, partition in partitions.items():
        output_path = OUTPUT_DIR / f"{name}.parquet"

        partition.to_parquet(
            output_path,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )

        split_metadata["partitions"][name] = {
            "rows": len(partition),
            "default_cases": int(partition[TARGET].sum()),
            "non_default_cases": int((partition[TARGET] == 0).sum()),
            "default_rate": float(partition[TARGET].mean()),
            "file": str(output_path),
        }

        print(f"\n{name.upper()}")
        print("Rows:", f"{len(partition):,}")
        print("Default cases:", f"{int(partition[TARGET].sum()):,}")
        print("Default rate:", f"{partition[TARGET].mean():.4%}")
        print("Saved:", output_path)

    metadata_path = OUTPUT_DIR / "split_metadata.json"

    metadata_path.write_text(
        json.dumps(split_metadata, indent=4),
        encoding="utf-8",
    )

    print("\nSplit metadata:", metadata_path)
    print("\nAPPLICATION DATA SPLITTING COMPLETED.")
    print("Customer ID overlap: None")
    print("Original rows preserved:", len(df))


if __name__ == "__main__":
    main()
