from pathlib import Path

import pandas as pd

source = Path("data/raw/paysim/PS_20174392719_1491204439457_log.csv")
out_dir = Path("reports/data_exploration")
out_dir.mkdir(parents=True, exist_ok=True)

parts = []

for chunk in pd.read_csv(source, chunksize=250_000):
    zero_rows = chunk.loc[chunk["amount"] == 0].copy()
    if not zero_rows.empty:
        parts.append(zero_rows)

if not parts:
    print("No zero-amount transactions found.")
else:
    zeros = pd.concat(parts, ignore_index=True)

    print("ZERO-AMOUNT TRANSACTIONS")
    print("=" * 55)
    print("Total:", len(zeros))

    print("\nCounts by type and fraud label:")
    print(zeros.groupby(["type", "isFraud", "isFlaggedFraud"]).size().rename("count").to_string())

    print("\nBalance changes:")
    zeros["origin_change"] = zeros["newbalanceOrig"] - zeros["oldbalanceOrg"]
    zeros["destination_change"] = zeros["newbalanceDest"] - zeros["oldbalanceDest"]

    print(
        zeros[
            [
                "step",
                "type",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "isFraud",
                "isFlaggedFraud",
                "origin_change",
                "destination_change",
            ]
        ].to_string(index=False)
    )

    zeros.to_csv(
        out_dir / "paysim_zero_amount_transactions.csv",
        index=False,
    )
    print("\nSaved zero-amount records for inspection.")
