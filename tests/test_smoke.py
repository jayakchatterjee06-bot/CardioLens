from fastapi.testclient import TestClient

from backend.main import app
from ml.config import CONSTANT_COLUMNS, LEAKAGE_COLUMNS, TARGETS
from ml.data_loader import load_raw_dataset


def test_health_lists_all_targets():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["targets"] == list(TARGETS)


def test_dataset_has_expected_target_columns():
    df = load_raw_dataset()
    assert df.shape == (303, 59)
    for spec in TARGETS.values():
        assert set(df[spec.column].unique()) >= {spec.positive_label}


def test_leakage_columns_cover_every_target():
    assert set(LEAKAGE_COLUMNS) == {"Cath", "LAD", "LCX", "RCA"}
    assert not set(LEAKAGE_COLUMNS) & set(CONSTANT_COLUMNS)
