# CardioLens

Interactive 3D cardiovascular risk visualisation: predicts overall CAD and LAD / LCX / RCA
stenosis from clinical data and maps vessel-level risk onto a 3D heart.
Multimodal AI Hackathon 2026, Track A.

> **Decision support / educational use only.** Not a substitute for formal diagnostic imaging.

## Structure (Plan or Workflow of the project made by claude)

```
CardioLens/
├── ml/                     # data, preprocessing, training, SHAP
│   ├── config.py           # targets, leakage columns, CV settings, paths
│   ├── features.py         # registry of the 54 inputs (kind, group, unit, levels) + leakage guard
│   ├── data_loader.py      # raw xlsx -> DataFrame (+ value fixes)
│   ├── preprocess.py       # ColumnTransformer built from the registry
│   ├── estimators.py       # candidate models + per-target pipeline
│   ├── evaluation.py       # repeated stratified CV, selection, out-of-fold predictions
│   ├── calibration.py      # Platt scaling
│   ├── risk_model.py       # deployable unit: pipeline + calibrator
│   ├── explain.py          # SHAP in probability units, folded back to clinical features
│   ├── artifacts.py        # save/load models, manifest, SHAP background
│   ├── reporting.py        # metrics.md/json + figures
│   ├── train.py            # entry point: python -m ml.train
│   ├── inspect_dataset.py  # dataset report
│   ├── data/raw/           # UCI #411 Extension of Z-Alizadeh Sani (xlsx + original zip)
│   ├── models/             # {cad,lad,lcx,rca}.joblib, manifest.json, shap_background.csv
│   └── reports/            # metrics.md, metrics.json, ROC / calibration / importance figures
├── backend/                # FastAPI (main.py, schemas.py, inference.py)
├── frontend/               # Vite + React + React Three Fiber
│   └── src/{components,api,config}
├── tools/                  # asset build scripts (heart_model.glb)
├── tests/                  # pytest smoke tests
├── docs/                   # submission documentation
├── requirements.txt        # one venv shared by ml/ and backend/
└── setup.sh                # one-shot macOS setup
```

## Setup 

```bash
./setup.sh
```
Installs Python 3.11 + Node via Homebrew if missing, creates `.venv`, installs Python deps
(freezing exact versions to `requirements.lock.txt`) and runs `npm install` in `frontend/`.

## Run

```bash
# terminal 1 — API on :8000
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# terminal 2 — UI on :5173 (proxies /api to :8000)
cd frontend && npm run dev
```

Other commands (from the project root, venv active):
```bash
python -m ml.train             # retrain all 4 models + regenerate ml/reports (~2 min)
python -m ml.inspect_dataset   # dataset report
pytest -q tests                # test suite
```

## Models

For each target (CAD, LAD, LCX, RCA), Logistic Regression, Random Forest and Gradient Boosting are
compared under repeated stratified 5-fold CV (5 repeats). The model with the highest ROC-AUC is
refitted on all 303 patients, then wrapped with a Platt calibrator fitted on its out-of-fold
predictions so that predicted probabilities match observed frequencies (they drive the 3D colours).
Each target is saved as one self-contained `RiskModel` (preprocessing + classifier + calibrator).

| Target | Model | ROC-AUC (CV) | Brier (raw → calibrated) |
|---|---|---|---|
| Overall CAD | Random Forest | 0.923 ± 0.037 | 0.116 → 0.100 |
| LAD | Random Forest | 0.852 ± 0.047 | 0.164 → 0.152 |
| LCX | Random Forest | 0.729 ± 0.065 | 0.207 → 0.206 |
| RCA | Logistic Regression | 0.728 ± 0.055 | 0.214 → 0.205 |

Full tables, per-model comparisons and figures: `ml/reports/`. SHAP explanations are reported in
probability units (a contribution of +0.12 = +12 percentage points of risk) for every model type.

## Dataset notes (UCI #411, 303 rows × 59 columns, no missing values)

| Target | Column | Positive label | Positives |
|---|---|---|---|
| Overall CAD | `Cath` | `CAD` | 216 (71.3%) |
| LAD stenosis | `LAD` | `Stenotic` | 177 (58.4%) |
| LCX stenosis | `LCX` | `Stenotic` | 119 (39.3%) |
| RCA stenosis | `RCA` | `Stenotic` | 114 (37.6%) |

- `Cath`, `LAD`, `LCX`, `RCA` are excluded from the inputs of **every** model (leakage rule).
- `Cath == CAD` matches "any vessel stenotic" on 302/303 rows.
- `Exertional CP` is constant (all `N`) and is dropped; 54 input features remain.
- `Sex` has a typo (`Fmale`), normalised to `Female` at load time.
- `Length` is height in cm. The workbook's `Sheet1` is an empty template and is ignored.

## 3D heart model

`frontend/public/heart_model.glb` is built from **BodyParts3D** (© The Database Center for Life
Science, licensed under CC BY 4.0) by `tools/build_heart_model.py`. The script downloads the
BodyParts3D part-of meshes and groups them by FMA concept into named nodes:

| Node | FMA source | Driven by |
|---|---|---|
| `lad` | FMA3862 anterior interventricular branch of LCA | LAD model |
| `lcx` | FMA3895 circumflex branch of LCA | LCX model |
| `rca` | FMA50039 right coronary artery | RCA model |
| `left_main`, `myocardium`, `aorta`, `pulmonary_trunk`, `coronary_veins` | FMA3855, FMA7088, FMA3736+3768, FMA8612, coronary veins | context only |

The vessel node names are the same keys the ML targets and the frontend use, so a prediction
for `lad` can only ever colour the LAD mesh. About 66k triangles and 1.2 MB, so it runs on
integrated graphics.

Rebuild (only needed if the script changes):
```bash
pip install -r tools/requirements.txt
python -m tools.build_heart_model
```
