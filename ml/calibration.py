"""Post-hoc Platt scaling of predicted probabilities.

Tree ensembles such as Random Forest compress probabilities toward 0.5. The 3D viewer colours
vessels by probability band, so probabilities must match observed frequencies. A two-parameter
logistic map on the logit of the model output is fitted to out-of-fold predictions; being
monotone, it leaves ranking (and therefore ROC-AUC) unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ml.config import CV_SPLITS, RANDOM_STATE

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


@dataclass(frozen=True)
class PlattCalibrator:
    slope: float = 1.0
    intercept: float = 0.0

    @classmethod
    def fit(cls, probabilities: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        model = LogisticRegression(C=1e6).fit(_logit(probabilities).reshape(-1, 1), y)
        return cls(slope=float(model.coef_[0, 0]), intercept=float(model.intercept_[0]))

    def __call__(self, probabilities: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.slope * _logit(probabilities) + self.intercept)))

    def derivative(self, probability: float) -> float:
        """d calibrated / d raw at one point (used when a patient sits exactly at the base value)."""
        p = float(np.clip(probability, _EPS, 1 - _EPS))
        q = float(self(np.array([p]))[0])
        return self.slope * q * (1 - q) / (p * (1 - p))

    def to_dict(self) -> dict:
        return {"method": "platt", "slope": round(self.slope, 5), "intercept": round(self.intercept, 5)}


def cross_fitted_calibration(probabilities: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calibrated probabilities where each patient's calibrator was fitted without that patient."""
    probabilities, y = np.asarray(probabilities), np.asarray(y)
    calibrated = np.empty_like(probabilities, dtype=float)
    folds = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train, test in folds.split(probabilities.reshape(-1, 1), y):
        calibrated[test] = PlattCalibrator.fit(probabilities[train], y[train])(probabilities[test])
    return calibrated
