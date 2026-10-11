"""Reading and writing trained artifacts. Shared by training and the backend."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from ml.config import MODELS_DIR
from ml.features import FEATURE_NAMES, assert_no_leakage
from ml.risk_model import RiskModel

MANIFEST_FILE = "manifest.json"
BACKGROUND_FILE = "shap_background.csv"


def model_path(target_key: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / f"{target_key}.joblib"


def save_model(target_key: str, model: RiskModel, models_dir: Path = MODELS_DIR) -> Path:
    assert_no_leakage(model.feature_names_in_)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = model_path(target_key, models_dir)
    joblib.dump(model, path)
    return path


def load_model(target_key: str, models_dir: Path = MODELS_DIR) -> RiskModel:
    model = joblib.load(model_path(target_key, models_dir))
    if not isinstance(model, RiskModel):
        raise TypeError(f"{target_key}: expected RiskModel, found {type(model).__name__}")
    assert_no_leakage(model.feature_names_in_)
    return model


def save_background(background: pd.DataFrame, models_dir: Path = MODELS_DIR) -> Path:
    path = models_dir / BACKGROUND_FILE
    background.loc[:, list(FEATURE_NAMES)].to_csv(path, index=False)
    return path


def load_background(models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    return pd.read_csv(models_dir / BACKGROUND_FILE, keep_default_na=False)


def write_manifest(manifest: dict, models_dir: Path = MODELS_DIR) -> Path:
    path = models_dir / MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return path


def read_manifest(models_dir: Path = MODELS_DIR) -> dict:
    return json.loads((models_dir / MANIFEST_FILE).read_text())
