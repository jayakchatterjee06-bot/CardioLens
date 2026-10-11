"""Single source of truth for dataset paths, target definitions and leakage rules.

Every other module (training, inference, API, docs) reads from here so the
target/feature contract can never drift between training time and inference time.
"""
from dataclasses import dataclass
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ML_ROOT / "data" / "raw" / "z_alizadeh_sani_extension.xlsx"
RAW_SHEET_NAME = "Sheet 1 - Table 1"  # "Sheet1" in the workbook is an empty header-only template
PROCESSED_DIR = ML_ROOT / "data" / "processed"
MODELS_DIR = ML_ROOT / "models"
REPORTS_DIR = ML_ROOT / "reports"

RANDOM_STATE = 42

# Validation: repeated stratified k-fold. With 303 rows a single hold-out set (~60 patients)
# is too noisy, so every patient is tested once per repeat and metrics are reported as mean ± std.
CV_SPLITS = 5
CV_REPEATS = 5
SELECTION_METRIC = "roc_auc"
DECISION_THRESHOLD = 0.5

# Rows kept as the SHAP reference distribution (saved alongside the models).
SHAP_BACKGROUND_SIZE = 100


@dataclass(frozen=True)
class TargetSpec:
    """Describes one prediction task: which column, and which label counts as positive."""

    key: str            # stable id used by the API and the 3D viewer (e.g. "lad")
    column: str         # column name in the raw dataset
    positive_label: str  # raw value that maps to class 1
    display_name: str


TARGETS: dict[str, TargetSpec] = {
    "cad": TargetSpec("cad", "Cath", "CAD", "Overall CAD"),
    "lad": TargetSpec("lad", "LAD", "Stenotic", "Left Anterior Descending"),
    "lcx": TargetSpec("lcx", "LCX", "Stenotic", "Left Circumflex"),
    "rca": TargetSpec("rca", "RCA", "Stenotic", "Right Coronary Artery"),
}

# Requirement 1d: none of these may ever be a model input, for ANY target.
LEAKAGE_COLUMNS: tuple[str, ...] = tuple(spec.column for spec in TARGETS.values())

# Zero-variance in this dataset (all 303 rows are "N"), so it carries no signal.
CONSTANT_COLUMNS: tuple[str, ...] = ("Exertional CP",)

# Known data-entry typos in the raw file, normalised at load time.
VALUE_FIXES: dict[str, dict[str, str]] = {
    "Sex": {"Fmale": "Female"},
}
