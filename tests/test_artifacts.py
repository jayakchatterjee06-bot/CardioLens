"""Checks the trained artifacts in ml/models (skipped until `python -m ml.train` has run)."""
import pytest

from ml.artifacts import load_background, load_model, read_manifest
from ml.config import LEAKAGE_COLUMNS, MODELS_DIR, TARGETS
from ml.features import FEATURE_NAMES

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "manifest.json").exists(), reason="models not trained yet")


def test_manifest_covers_every_target():
    manifest = read_manifest()
    assert set(manifest["targets"]) == set(TARGETS)
    assert [f["name"] for f in manifest["features"]] == list(FEATURE_NAMES)


@pytest.mark.parametrize("key", list(TARGETS))
def test_saved_model_never_sees_target_columns(key, X):
    model = load_model(key)
    assert list(model.feature_names_in_) == list(FEATURE_NAMES)
    assert not set(model.feature_names_in_) & set(LEAKAGE_COLUMNS)
    probabilities = model.predict_proba(X)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_background_round_trips_with_original_dtypes(X):
    background = load_background()
    assert list(background.columns) == list(FEATURE_NAMES)
    load_model("cad").predict_proba(background)  # would raise on unknown categories
