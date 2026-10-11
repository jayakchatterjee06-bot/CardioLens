"""Task #2: print dataset structure and confirm target/leakage column names.

Run from the project root:  python -m ml.inspect_dataset
"""
import pandas as pd

from ml.config import CONSTANT_COLUMNS, LEAKAGE_COLUMNS, TARGETS
from ml.data_loader import binary_target, load_raw_dataset


def main() -> None:
    df = load_raw_dataset()
    print(f"Shape: {df.shape}\n")
    df.info()
    print("\nNumeric summary:")
    print(df.describe().T.round(2).to_string())

    missing = [c for c in LEAKAGE_COLUMNS + CONSTANT_COLUMNS if c not in df.columns]
    assert not missing, f"Expected columns not found: {missing}"

    print("\nTarget balance (positive rate):")
    for key, spec in TARGETS.items():
        y = binary_target(df, key)
        print(f"  {key:<4} column={spec.column!r:<7} positives={y.sum():>3}/{len(y)} ({y.mean():.1%})")

    any_vessel = pd.concat([binary_target(df, k) for k in ("lad", "lcx", "rca")], axis=1).max(axis=1)
    agreement = (any_vessel == binary_target(df, "cad")).mean()
    print(f"\nCath==CAD agrees with 'any vessel stenotic' on {agreement:.1%} of rows")

    features = [c for c in df.columns if c not in LEAKAGE_COLUMNS + CONSTANT_COLUMNS]
    print(f"\nModel input features ({len(features)}): {features}")


if __name__ == "__main__":
    main()
