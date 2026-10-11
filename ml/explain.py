"""SHAP explanations for any trained per-target pipeline.

Every model is explained in the same unit, probability of the positive class, so a
contribution of +0.12 always means "this feature raised the predicted risk by 12 percentage
points" regardless of which model won for that target. SHAP values for encoded columns are
folded back into the 54 clinical features, and raw (unscaled) patient values are reported.

Calibration: SHAP is computed on the classifier's own probability, then each patient's
contributions are rescaled by (calibrated change) / (raw change) relative to the base value.
This keeps signs and ranking, and contributions still sum exactly to the calibrated probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier

from ml.config import RANDOM_STATE
from ml.estimators import CLASSIFIER_STEP, PREPROCESS_STEP
from ml.features import FEATURE_NAMES, FEATURES_BY_NAME, select_features
from ml.preprocess import encoded_feature_sources
from ml.risk_model import RiskModel

TREE_MODELS = (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    DecisionTreeClassifier,
)


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    label: str
    group: str
    unit: str
    value: object
    contribution: float  # change in P(positive) attributed to this feature

    def to_dict(self) -> dict:
        value = self.value.item() if isinstance(self.value, np.generic) else self.value
        return {
            "feature": self.feature,
            "label": self.label,
            "group": self.group,
            "unit": self.unit,
            "value": value,
            "contribution": round(float(self.contribution), 5),
        }


@dataclass(frozen=True)
class Explanation:
    probability: float
    base_value: float  # average predicted probability over the background population
    contributions: tuple[FeatureContribution, ...]  # sorted by |contribution|, largest first

    def top(self, n: int) -> tuple[FeatureContribution, ...]:
        return self.contributions[:n]


class ShapBackend(Protocol):
    """Maps encoded rows to (SHAP values in probability units, base value)."""

    def __call__(self, encoded: np.ndarray) -> tuple[np.ndarray, float]: ...


class TreeShapBackend:
    """Exact interventional TreeSHAP, reported on the probability scale."""

    def __init__(self, model, background: np.ndarray) -> None:
        self._explainer = shap.TreeExplainer(
            model, data=background, model_output="probability", feature_perturbation="interventional"
        )

    def __call__(self, encoded: np.ndarray) -> tuple[np.ndarray, float]:
        result = self._explainer(encoded, check_additivity=False)
        values, base = np.asarray(result.values), np.asarray(result.base_values)
        if values.ndim == 3:  # one output per class -> keep the positive class
            values, base = values[..., 1], base[..., 1]
        return values, float(np.ravel(base)[0])


class PermutationShapBackend:
    """Model-agnostic SHAP on predict_proba; used for non-tree models such as logistic regression."""

    def __init__(self, model, background: np.ndarray) -> None:
        self._explainer = shap.PermutationExplainer(
            lambda rows: model.predict_proba(rows)[:, 1],
            shap.maskers.Independent(background, max_samples=len(background)),
            seed=RANDOM_STATE,
        )

    def __call__(self, encoded: np.ndarray) -> tuple[np.ndarray, float]:
        result = self._explainer(encoded, max_evals=2 * encoded.shape[1] + 1 + 400, silent=True)
        return np.asarray(result.values), float(np.ravel(result.base_values)[0])


def make_backend(model, background: np.ndarray) -> ShapBackend:
    if isinstance(model, TREE_MODELS):
        return TreeShapBackend(model, background)
    return PermutationShapBackend(model, background)


class RiskModelExplainer:
    """Explains one trained RiskModel in clinical-feature terms, on the calibrated probability scale."""

    def __init__(self, model: RiskModel, background: pd.DataFrame) -> None:
        self._model = model
        self._preprocess = model.pipeline.named_steps[PREPROCESS_STEP]
        classifier = model.pipeline.named_steps[CLASSIFIER_STEP]
        self._fold = self._folding_matrix(encoded_feature_sources(self._preprocess))
        encoded_background = self._preprocess.transform(select_features(background))
        self._backend = make_backend(classifier, encoded_background)

    @staticmethod
    def _folding_matrix(sources: list[str]) -> np.ndarray:
        """(n_encoded, n_features) 0/1 matrix that sums encoded SHAP columns per clinical feature."""
        index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        fold = np.zeros((len(sources), len(FEATURE_NAMES)))
        for row, source in enumerate(sources):
            fold[row, index[source]] = 1.0
        return fold

    def feature_shap(self, patients: pd.DataFrame) -> tuple[np.ndarray, float]:
        """SHAP matrix (n_patients, 54) on the calibrated probability scale, plus the base value."""
        rows = select_features(patients)
        raw_values, raw_base = self._backend(self._preprocess.transform(rows))
        raw_values = raw_values @ self._fold
        return self._to_calibrated_scale(raw_values, self._model.raw_probability(rows), raw_base)

    def _to_calibrated_scale(self, values: np.ndarray, raw_pred: np.ndarray, raw_base: float) -> tuple[np.ndarray, float]:
        calibrate = self._model.calibrator
        base = float(calibrate(np.array([raw_base]))[0])
        raw_delta = raw_pred - raw_base
        calibrated_delta = calibrate(raw_pred) - base
        near_base = np.abs(raw_delta) < 1e-9
        scale = np.where(near_base, calibrate.derivative(raw_base), calibrated_delta / np.where(near_base, 1.0, raw_delta))
        return values * scale[:, None], base

    def explain(self, patient: pd.DataFrame) -> Explanation:
        if len(patient) != 1:
            raise ValueError("explain() takes exactly one patient row; use feature_shap() for batches")
        row = select_features(patient)
        shap_row, base = self.feature_shap(row)
        probability = float(self._model.predict_proba(row)[0])

        contributions = [
            FeatureContribution(
                feature=name,
                label=FEATURES_BY_NAME[name].label,
                group=FEATURES_BY_NAME[name].group.value,
                unit=FEATURES_BY_NAME[name].unit,
                value=row.iloc[0][name],
                contribution=float(shap_row[0, i]),
            )
            for i, name in enumerate(FEATURE_NAMES)
        ]
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
        return Explanation(probability=probability, base_value=base, contributions=tuple(contributions))


def global_importance(explainer: RiskModelExplainer, patients: pd.DataFrame) -> list[dict]:
    """Mean |SHAP| per clinical feature across a population, largest first."""
    values, _ = explainer.feature_shap(patients)
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [
        {"feature": FEATURE_NAMES[i], "label": FEATURES_BY_NAME[FEATURE_NAMES[i]].label, "mean_abs_shap": round(float(mean_abs[i]), 5)}
        for i in order
    ]
