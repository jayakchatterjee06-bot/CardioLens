import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from ml.calibration import PlattCalibrator, cross_fitted_calibration


@pytest.fixture
def squashed():
    """True risk spread over 0..1, but the 'model' reports it compressed toward 0.5 (like a random forest)."""
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.02, 0.98, 2000)
    y = (rng.uniform(size=true_p.size) < true_p).astype(int)
    return 0.5 + (true_p - 0.5) * 0.4, y


def test_identity_calibrator_changes_nothing():
    p = np.linspace(0.01, 0.99, 50)
    assert PlattCalibrator()(p) == pytest.approx(p)


def test_calibration_is_monotone_so_auc_is_unchanged(squashed):
    p, y = squashed
    calibrated = PlattCalibrator.fit(p, y)(p)
    assert np.all(np.diff(calibrated[np.argsort(p)]) >= 0)
    assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, p))


def test_cross_fitted_calibration_improves_brier_on_compressed_probabilities(squashed):
    p, y = squashed
    assert brier_score_loss(y, cross_fitted_calibration(p, y)) < brier_score_loss(y, p)


def test_derivative_matches_finite_difference():
    calibrator = PlattCalibrator(slope=1.7, intercept=-0.3)
    h = 1e-6
    numeric = (calibrator(np.array([0.4 + h]))[0] - calibrator(np.array([0.4 - h]))[0]) / (2 * h)
    assert calibrator.derivative(0.4) == pytest.approx(numeric, rel=1e-4)
