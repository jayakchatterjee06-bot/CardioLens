"""Candidate classifiers and the per-target pipeline that wraps them.

Hyperparameters are fixed, conservative defaults suited to ~300 rows. They are not tuned
per target, which keeps model selection to a single comparison (3 candidates) and limits
optimistic bias in the reported cross-validation scores.

No class re-weighting is applied: imbalance is mild (38-71% positive) and re-weighting would
shift predicted probabilities away from observed frequencies, and those probabilities drive
the risk colours in the 3D viewer.
"""
from __future__ import annotations

from collections.abc import Callable

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.config import RANDOM_STATE
from ml.preprocess import build_preprocessor

PREPROCESS_STEP = "preprocess"
CLASSIFIER_STEP = "classifier"

CANDIDATES: dict[str, Callable[[], ClassifierMixin]] = {
    "logistic_regression": lambda: LogisticRegression(C=0.5, max_iter=5000),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=1,  # parallelism happens across CV folds instead
    ),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_depth=3,
        min_samples_leaf=10,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    ),
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def build_pipeline(model_name: str) -> Pipeline:
    if model_name not in CANDIDATES:
        raise KeyError(f"Unknown model {model_name!r}; choose from {sorted(CANDIDATES)}")
    return Pipeline([
        (PREPROCESS_STEP, build_preprocessor()),
        (CLASSIFIER_STEP, CANDIDATES[model_name]()),
    ])
