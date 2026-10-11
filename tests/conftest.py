import pandas as pd
import pytest

from ml.data_loader import load_raw_dataset
from ml.features import select_features


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return load_raw_dataset()


@pytest.fixture(scope="session")
def X(raw_df) -> pd.DataFrame:
    return select_features(raw_df)
