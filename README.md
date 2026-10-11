# CardioLens

Interactive 3D cardiovascular risk visualisation: predicts overall CAD and LAD / LCX / RCA
stenosis from clinical data and maps vessel-level risk onto a 3D heart.
Multimodal AI Hackathon 2026, Track A.

> **Decision support / educational use only.** Not a substitute for formal diagnostic imaging.

## Structure (Plan or Workflow of the project made by claude)

```
CardioLens/
├── ml/                     # data, preprocessing, training, SHAP
│   ├── config.py           # targets, leakage columns, paths (single source of truth)
│   ├── data_loader.py      # raw xlsx -> DataFrame (+ value fixes)
│   ├── inspect_dataset.py  # Task #2 dataset report
│   ├── preprocess.py       # Task #3
│   ├── train.py            # Task #4
│   ├── explain.py          # Task #5
│   ├── data/raw/           # UCI #411 Extension of Z-Alizadeh Sani (xlsx + original zip)
│   └── models/             # saved pipelines (.joblib)
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
python -m ml.inspect_dataset   # dataset report
pytest -q tests                # smoke tests
```

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
