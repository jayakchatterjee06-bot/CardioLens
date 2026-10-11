"""End-to-end API tests against the real trained models (skipped until `python -m ml.train` has run)."""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import PatientInput
from ml.config import LEAKAGE_COLUMNS, MODELS_DIR, TARGETS
from ml.features import FEATURE_NAMES

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "manifest.json").exists(), reason="models not trained yet")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:  # runs the lifespan, i.e. real model loading
        yield test_client


@pytest.fixture(scope="module")
def samples(client):
    return {s["id"]: s for s in client.get("/api/samples").json()}


@pytest.fixture(scope="module")
def predictions(client, samples):
    return {sid: client.post("/api/predict", json=s["patient"]).json()["predictions"] for sid, s in samples.items()}


def probability(predictions, sample_id, target):
    return predictions[sample_id][target]["probability"]


def test_health_reports_models_loaded(client):
    assert client.get("/api/health").json() == {"status": "ok", "models_loaded": True, "targets": list(TARGETS)}


def test_metadata_exposes_the_registry_and_model_metrics(client):
    meta = client.get("/api/metadata").json()
    assert [f["name"] for f in meta["features"]] == list(FEATURE_NAMES)
    assert all(f["min"] is not None for f in meta["features"] if f["kind"] == "continuous")
    assert {t["key"] for t in meta["targets"]} == set(TARGETS)
    assert all(0.5 < t["cv_metrics"]["roc_auc"]["mean"] <= 1 for t in meta["targets"])


def test_sample_ground_truth_matches_their_description(samples):
    assert not any(samples["no-cad"]["ground_truth"].values())
    assert samples["lad-only"]["ground_truth"] == {"cad": True, "lad": True, "lcx": False, "rca": False}
    assert all(samples["three-vessel"]["ground_truth"].values())


def test_every_sample_is_a_valid_request(samples):
    for sample in samples.values():
        PatientInput.model_validate(sample["patient"])


def test_prediction_covers_all_targets_and_contributions_add_up(predictions):
    for result in predictions.values():
        assert set(result) == set(TARGETS)
        for target in result.values():
            assert 0 <= target["probability"] <= 1
            assert len(target["contributions"]) == len(FEATURE_NAMES)
            total = target["base_value"] + sum(c["contribution"] for c in target["contributions"])
            assert total == pytest.approx(target["probability"], abs=5e-3)


def test_risk_follows_disease_burden(predictions):
    assert probability(predictions, "three-vessel", "cad") > probability(predictions, "no-cad", "cad")
    assert probability(predictions, "lad-only", "lad") > probability(predictions, "no-cad", "lad")
    for vessel in ("lad", "lcx", "rca"):
        assert probability(predictions, "three-vessel", vessel) > probability(predictions, "no-cad", vessel)


def test_lad_only_patient_is_flagged_on_the_lad(predictions):
    lad = probability(predictions, "lad-only", "lad")
    assert lad > probability(predictions, "lad-only", "lcx")
    assert lad > probability(predictions, "lad-only", "rca")


def test_rejects_a_missing_feature(client, samples):
    payload = dict(samples["no-cad"]["patient"])
    del payload["Age"]
    assert client.post("/api/predict", json=payload).status_code == 422


@pytest.mark.parametrize("column", LEAKAGE_COLUMNS)
def test_rejects_target_columns_in_the_payload(client, samples, column):
    payload = {**samples["no-cad"]["patient"], column: "Normal"}
    assert client.post("/api/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field, value", [("Age", 5), ("EF-TTE", 120), ("Sex", "M"), ("VHD", "severe"), ("DM", 2)])
def test_rejects_implausible_or_undeclared_values(client, samples, field, value):
    payload = {**samples["no-cad"]["patient"], field: value}
    assert client.post("/api/predict", json=payload).status_code == 422


def test_prediction_routes_answer_503_without_models(client, samples):
    service, app.state.service = app.state.service, None
    try:
        assert client.post("/api/predict", json=samples["no-cad"]["patient"]).status_code == 503
        assert client.get("/api/metadata").status_code == 503
        assert client.get("/api/health").json()["models_loaded"] is False
    finally:
        app.state.service = service
