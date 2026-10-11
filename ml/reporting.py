"""Writes the evaluation report (metrics.json, metrics.md) and figures for the documentation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from ml.calibration import PlattCalibrator  # noqa: E402
from ml.config import CV_REPEATS, CV_SPLITS, DECISION_THRESHOLD, REPORTS_DIR, TargetSpec  # noqa: E402
from ml.estimators import DISPLAY_NAMES  # noqa: E402
from ml.evaluation import METRICS, CVResult, probability_metrics  # noqa: E402

# Validated categorical order (blue, orange, aqua, yellow); each target keeps its colour in every figure.
TARGET_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


@dataclass
class TargetReport:
    spec: TargetSpec
    candidates: list[CVResult]
    selected: CVResult
    calibrator: PlattCalibrator
    y_true: np.ndarray
    oof_raw: np.ndarray         # out-of-fold P(positive) from the selected model
    oof_calibrated: np.ndarray  # same, after cross-fitted Platt scaling
    importance: list[dict]

    def raw_metrics(self) -> dict[str, float]:
        return probability_metrics(self.y_true, self.oof_raw, DECISION_THRESHOLD)

    def final_metrics(self) -> dict[str, float]:
        return probability_metrics(self.y_true, self.oof_calibrated, DECISION_THRESHOLD)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _fmt(summary) -> str:
    return f"{summary.mean:.3f} ± {summary.std:.3f}"


def write_metrics_json(reports: list[TargetReport], path: Path) -> None:
    payload = {
        r.spec.key: {
            "target_column": r.spec.column,
            "selected_model": r.selected.model_name,
            "calibration": r.calibrator.to_dict(),
            "oof_metrics_raw": r.raw_metrics(),
            "oof_metrics_calibrated": r.final_metrics(),
            "candidates": {c.model_name: c.to_dict() for c in r.candidates},
        }
        for r in reports
    }
    path.write_text(json.dumps(payload, indent=2))


def write_metrics_markdown(reports: list[TargetReport], path: Path) -> None:
    header = "| Model | " + " | ".join(m.replace("_", "-").upper() if m == "roc_auc" else m.capitalize() for m in METRICS) + " |"
    divider = "|" + "---|" * (len(METRICS) + 1)
    lines = [
        "# Model evaluation",
        "",
        f"Repeated stratified {CV_SPLITS}-fold cross-validation ({CV_REPEATS} repeats, {CV_SPLITS * CV_REPEATS} fits per model). "
        "Values are mean ± std across folds. Brier score: lower is better. ★ = selected (highest ROC-AUC).",
        "",
        "## Selected models",
        "",
        "| Target | Model | ROC-AUC | Accuracy | Precision | Recall | F1 | Brier |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        m = r.selected.metrics
        lines.append(
            f"| {r.spec.display_name} | {DISPLAY_NAMES[r.selected.model_name]} | {_fmt(m['roc_auc'])} | {_fmt(m['accuracy'])} | "
            f"{_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | {_fmt(m['brier'])} |"
        )
    lines += [
        "",
        "## Final deployed models (selected model + Platt calibration)",
        "",
        f"Out-of-fold predictions; the calibrator for each patient was fitted without that patient. Threshold {DECISION_THRESHOLD}.",
        "",
        "| Target | ROC-AUC | Accuracy | Precision | Recall | F1 | Brier (raw → calibrated) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        f, raw = r.final_metrics(), r.raw_metrics()
        lines.append(
            f"| {r.spec.display_name} | {f['roc_auc']:.3f} | {f['accuracy']:.3f} | {f['precision']:.3f} | "
            f"{f['recall']:.3f} | {f['f1']:.3f} | {raw['brier']:.3f} → {f['brier']:.3f} |"
        )
    for r in reports:
        lines += ["", f"## {r.spec.display_name} (`{r.spec.column}`)", "", header, divider]
        for c in r.candidates:
            star = " ★" if c.model_name == r.selected.model_name else ""
            lines.append(f"| {DISPLAY_NAMES[c.model_name]}{star} | " + " | ".join(_fmt(c.metrics[m]) for m in METRICS) + " |")
        top = ", ".join(f"{f['label']} ({f['mean_abs_shap']:.3f})" for f in r.importance[:5])
        lines += ["", f"Top features by mean |SHAP|: {top}"]
    path.write_text("\n".join(lines) + "\n")


def plot_roc(reports: list[TargetReport], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.2), dpi=160)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--", label="Chance")
    for r, colour in zip(reports, TARGET_COLOURS):
        fpr, tpr, _ = roc_curve(r.y_true, r.oof_raw)
        auc = roc_auc_score(r.y_true, r.oof_raw)
        ax.plot(fpr, tpr, color=colour, linewidth=2, label=f"{r.spec.display_name} (AUC {auc:.2f})")
    ax.set_xlabel("False positive rate", color=MUTED)
    ax.set_ylabel("True positive rate", color=MUTED)
    ax.set_title("ROC curves (out-of-fold predictions)", color=INK, fontsize=11, loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_calibration(reports: list[TargetReport], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.2), dpi=160, sharex=True, sharey=True)
    for ax, r, colour in zip(axes.flat, reports, TARGET_COLOURS):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
        for probabilities, style, alpha in ((r.oof_raw, "--", 0.45), (r.oof_calibrated, "-", 1.0)):
            observed, predicted = calibration_curve(r.y_true, probabilities, n_bins=5, strategy="quantile")
            ax.plot(predicted, observed, color=colour, linewidth=2, linestyle=style, marker="o", markersize=5, alpha=alpha)
        ax.set_title(f"{r.spec.display_name} · {DISPLAY_NAMES[r.selected.model_name]}", color=INK, fontsize=9, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("Predicted probability", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed frequency", color=MUTED, fontsize=9)
    fig.suptitle("Calibration, out-of-fold (5 quantile bins)", color=INK, fontsize=11, x=0.02, ha="left")
    handles = [
        plt.Line2D([], [], color=MUTED, linewidth=2, linestyle="--", alpha=0.6, marker="o", label="Model output"),
        plt.Line2D([], [], color=INK, linewidth=2, marker="o", label="After Platt calibration"),
        plt.Line2D([], [], color=AXIS, linewidth=1, linestyle="--", label="Perfect calibration"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_importance(reports: list[TargetReport], path: Path, top_n: int = 10) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), dpi=160)
    for ax, r, colour in zip(axes.flat, reports, TARGET_COLOURS):
        _style(ax)
        ax.grid(axis="y", visible=False)
        top = r.importance[:top_n][::-1]
        ax.barh([f["label"] for f in top], [f["mean_abs_shap"] for f in top], color=colour, height=0.6)
        ax.set_title(r.spec.display_name, color=INK, fontsize=10, loc="left")
        ax.tick_params(axis="y", colors=INK, labelsize=8)
        ax.set_xlabel("Mean |SHAP| (probability)", color=MUTED, fontsize=8)
    fig.suptitle(f"Top {top_n} features per target (global SHAP importance)", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_reports(reports: list[TargetReport], reports_dir: Path = REPORTS_DIR) -> list[Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics.json": write_metrics_json,
        "metrics.md": write_metrics_markdown,
        "roc_curves.png": plot_roc,
        "calibration.png": plot_calibration,
        "feature_importance.png": plot_importance,
    }
    paths = []
    for name, writer in outputs.items():
        path = reports_dir / name
        writer(reports, path)
        paths.append(path)
    return paths
