"""Prediction service: loads the trained artifacts once and serves predictions + SHAP explanations."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from backend.settings import DISCLAIMER
from ml.artifacts import MANIFEST_FILE, load_background, load_model, read_manifest
from ml.config import MODELS_DIR, TARGETS, TargetSpec
from ml.explain import RiskModelExplainer
from ml.features import FEATURE_NAMES, FEATURES, select_features
from ml.risk_model import RiskModel


class ModelsNotTrainedError(RuntimeError):
    """No trained artifacts found."""


class ArtifactMismatchError(RuntimeError):
    """Trained artifacts no longer match the code (feature registry or targets changed)."""


class Predictor(Protocol):
    def predict(self, record: Mapping[str, Any]) -> dict: ...

    def metadata(self) -> dict: ...


@dataclass(frozen=True)
class _TargetRuntime:
    spec: TargetSpec
    model_name: str
    model: RiskModel
    explainer: RiskModelExplainer


def _check_manifest_matches_code(manifest: dict) -> None:
    trained_features = [f["name"] for f in manifest["features"]]
    if trained_features != list(FEATURE_NAMES):
        raise ArtifactMismatchError("Feature registry changed since training; run `python -m ml.train`.")
    if set(manifest["targets"]) != set(TARGETS):
        raise ArtifactMismatchError("Targets changed since training; run `python -m ml.train`.")


class PredictionService:
    """Holds one calibrated model and one SHAP explainer per target. Build it once per process."""

    def __init__(self, manifest: dict, runtimes: dict[str, _TargetRuntime]) -> None:
        self._manifest = manifest
        self._runtimes = runtimes
        self._threshold = float(manifest["decision_threshold"])

    @classmethod
    def from_artifacts(cls, models_dir: Path = MODELS_DIR) -> PredictionService:
        if not (models_dir / MANIFEST_FILE).exists():
            raise ModelsNotTrainedError(f"No trained models in {models_dir}; run `python -m ml.train`.")
        manifest = read_manifest(models_dir)
        _check_manifest_matches_code(manifest)
        background = load_background(models_dir)

        runtimes = {}
        for key, spec in TARGETS.items():
            model = load_model(key, models_dir)
            runtimes[key] = _TargetRuntime(
                spec=spec,
                model_name=manifest["targets"][key]["model"],
                model=model,
                explainer=RiskModelExplainer(model, background),
            )
        return cls(manifest, runtimes)

    def predict(self, record: Mapping[str, Any]) -> dict:
        patient = select_features(pd.DataFrame([dict(record)]))
        return {
            "predictions": {key: self._predict_target(runtime, patient) for key, runtime in self._runtimes.items()},
            "disclaimer": DISCLAIMER,
        }

    def _predict_target(self, runtime: _TargetRuntime, patient: pd.DataFrame) -> dict:
        explanation = runtime.explainer.explain(patient)
        return {
            "target": runtime.spec.key,
            "display_name": runtime.spec.display_name,
            "model": runtime.model_name,
            "probability": explanation.probability,
            "predicted_positive": explanation.probability >= self._threshold,
            "threshold": self._threshold,
            "base_value": explanation.base_value,
            "contributions": [c.to_dict() for c in explanation.contributions],
        }

    def metadata(self) -> dict:
        manifest = self._manifest
        return {
            "trained_at": manifest["trained_at"],
            "dataset": manifest["dataset"],
            "validation": manifest["validation"],
            "decision_threshold": self._threshold,
            "targets": [
                {
                    "key": key,
                    "display_name": info["display_name"],
                    "column": info["column"],
                    "positive_label": info["positive_label"],
                    "model": info["model"],
                    "cv_metrics": info["cv_metrics"],
                    "oof_metrics_calibrated": info["oof_metrics_calibrated"],
                    "global_importance": info["global_importance"],
                }
                for key, info in manifest["targets"].items()
            ],
            # Served from the live registry so labels/bounds stay current; names are checked against the manifest.
            "features": [f.to_dict() for f in FEATURES],
            "disclaimer": DISCLAIMER,
        }
