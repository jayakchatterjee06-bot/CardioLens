"""Cross-validated evaluation and model selection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate

from ml.config import CV_REPEATS, CV_SPLITS, RANDOM_STATE, SELECTION_METRIC
from ml.estimators import build_pipeline

# metric name -> (sklearn scorer, sign flip so that every stored value is "natural")
_SCORERS: dict[str, tuple[str, float]] = {
    "accuracy": ("accuracy", 1.0),
    "precision": ("precision", 1.0),
    "recall": ("recall", 1.0),
    "f1": ("f1", 1.0),
    "roc_auc": ("roc_auc", 1.0),
    "brier": ("neg_brier_score", -1.0),  # lower is better; measures probability calibration
}
METRICS: tuple[str, ...] = tuple(_SCORERS)


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float

    def to_dict(self) -> dict:
        return {"mean": round(self.mean, 4), "std": round(self.std, 4)}


@dataclass(frozen=True)
class CVResult:
    model_name: str
    metrics: dict[str, MetricSummary]

    def score(self, metric: str = SELECTION_METRIC) -> float:
        return self.metrics[metric].mean

    def to_dict(self) -> dict:
        return {name: summary.to_dict() for name, summary in self.metrics.items()}


def repeated_cv() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)


def cross_validate_model(model_name: str, X: pd.DataFrame, y: pd.Series) -> CVResult:
    scoring = {name: scorer for name, (scorer, _) in _SCORERS.items()}
    raw = cross_validate(build_pipeline(model_name), X, y, cv=repeated_cv(), scoring=scoring, n_jobs=-1)
    metrics = {}
    for name, (_, sign) in _SCORERS.items():
        values = sign * np.asarray(raw[f"test_{name}"])
        metrics[name] = MetricSummary(float(values.mean()), float(values.std()))
    return CVResult(model_name, metrics)


def select_best(results: list[CVResult], metric: str = SELECTION_METRIC) -> CVResult:
    return max(results, key=lambda r: r.score(metric))


def out_of_fold_probabilities(model_name: str, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """P(positive) for every patient from a model that never saw that patient (for ROC/calibration plots)."""
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(build_pipeline(model_name), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def probability_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Metrics for one set of (out-of-fold) probabilities at a fixed decision threshold."""
    predicted = (probability >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "precision": round(float(precision_score(y, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y, predicted)), 4),
        "f1": round(float(f1_score(y, predicted)), 4),
        "roc_auc": round(float(roc_auc_score(y, probability)), 4),
        "brier": round(float(brier_score_loss(y, probability)), 4),
    }
