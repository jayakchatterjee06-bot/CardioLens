"""Loads the raw Z-Alizadeh Sani extension dataset and applies value-level fixes only.

Feature engineering, encoding and scaling belong in preprocess.py, not here.
"""
import pandas as pd

from ml.config import RAW_DATA_PATH, RAW_SHEET_NAME, TARGETS, VALUE_FIXES


def load_raw_dataset(path=RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=RAW_SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]
    for column, mapping in VALUE_FIXES.items():
        if column in df.columns:
            df[column] = df[column].replace(mapping)
    return df


def binary_target(df: pd.DataFrame, target_key: str) -> pd.Series:
    """Returns the 0/1 label series for a target key ("cad", "lad", "lcx", "rca")."""
    spec = TARGETS[target_key]
    return (df[spec.column] == spec.positive_label).astype(int).rename(target_key)
