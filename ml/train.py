"""Trains the CAD, LAD, LCX and RCA models.

For each target: compare the candidate models with repeated stratified CV, select the best
by ROC-AUC, refit it on all patients, and save the full pipeline (preprocessing + model).
Writes ml/models/{target}.joblib, ml/models/manifest.json, ml/models/shap_background.csv
and the evaluation report in ml/reports/.

Run from the project root:  python -m ml.train
"""
from __future__ import annotations

import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import shap
import sklearn

from ml.artifacts import model_path, save_background, save_model, write_manifest
from ml.calibration import PlattCalibrator, cross_fitted_calibration
from ml.config import (
    CV_REPEATS,
    CV_SPLITS,
    DECISION_THRESHOLD,
    RANDOM_STATE,
    SELECTION_METRIC,
    SHAP_BACKGROUND_SIZE,
    TARGETS,
    TargetSpec,
)
from ml.data_loader import binary_target, load_raw_dataset
from ml.estimators import CANDIDATES, DISPLAY_NAMES, build_pipeline
from ml.evaluation import cross_validate_model, out_of_fold_probabilities, select_best
from ml.explain import RiskModelExplainer, global_importance
from ml.features import FEATURES, select_features
from ml.reporting import TargetReport, write_reports
from ml.risk_model import RiskModel


def train_target(spec: TargetSpec, X: pd.DataFrame, y: pd.Series, background: pd.DataFrame) -> TargetReport:
    candidates = [cross_validate_model(name, X, y) for name in CANDIDATES]
    selected = select_best(candidates)

    pipeline = build_pipeline(selected.model_name).fit(X, y)
    oof_raw = out_of_fold_probabilities(selected.model_name, X, y)
    model = RiskModel(pipeline=pipeline, calibrator=PlattCalibrator.fit(oof_raw, y.to_numpy()))
    save_model(spec.key, model)

    return TargetReport(
        spec=spec,
        candidates=candidates,
        selected=selected,
        calibrator=model.calibrator,
        y_true=y.to_numpy(),
        oof_raw=oof_raw,
        oof_calibrated=cross_fitted_calibration(oof_raw, y.to_numpy()),
        importance=global_importance(RiskModelExplainer(model, background), X),
    )


def build_manifest(reports: list[TargetReport], n_rows: int) -> dict:
    return {
        "schema_version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
        },
        "dataset": {"name": "UCI Extension of Z-Alizadeh Sani (id 411)", "rows": n_rows},
        "validation": {
            "scheme": "RepeatedStratifiedKFold",
            "n_splits": CV_SPLITS,
            "n_repeats": CV_REPEATS,
            "selection_metric": SELECTION_METRIC,
            "random_state": RANDOM_STATE,
        },
        "decision_threshold": DECISION_THRESHOLD,
        "targets": {
            r.spec.key: {
                "display_name": r.spec.display_name,
                "column": r.spec.column,
                "positive_label": r.spec.positive_label,
                "model": r.selected.model_name,
                "artifact": model_path(r.spec.key).name,
                "calibration": r.calibrator.to_dict(),
                "cv_metrics": r.selected.to_dict(),
                "oof_metrics_calibrated": r.final_metrics(),
                "global_importance": r.importance[:15],
            }
            for r in reports
        },
        "features": [f.to_dict() for f in FEATURES],
    }


def main() -> None:
    started = time.perf_counter()
    df = load_raw_dataset()
    X = select_features(df)
    background = X.sample(n=SHAP_BACKGROUND_SIZE, random_state=RANDOM_STATE)
    save_background(background)

    reports = []
    for key, spec in TARGETS.items():
        y = binary_target(df, key)
        report = train_target(spec, X, y, background)
        reports.append(report)
        auc = report.selected.metrics["roc_auc"]
        brier_raw, brier_cal = report.raw_metrics()["brier"], report.final_metrics()["brier"]
        print(
            f"{spec.display_name:<26} -> {DISPLAY_NAMES[report.selected.model_name]:<20} "
            f"ROC-AUC {auc.mean:.3f} ± {auc.std:.3f} | Brier {brier_raw:.3f} -> {brier_cal:.3f} calibrated",
            flush=True,
        )

    write_manifest(build_manifest(reports, len(df)))
    for path in write_reports(reports):
        print(f"  wrote {path.relative_to(path.parents[2])}")
    print(f"Done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
