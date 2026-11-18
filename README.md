# PathSnap — GPS Map-Matching & Route Reconstruction

Noisy GPS trajectories → most likely road segment per observation, using an ML candidate classifier
followed by HMM/Viterbi decoding, served by a FastAPI backend (Laptop 1) to an evaluator UI (Laptop 2).

> Work in progress — full install, reproduction and deployment instructions will be added before submission.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .            # makes `import pathsnap` work from anywhere
```
Unzip the organiser data package so that `data/raw/train/` and `data/raw/student_test/` exist (not committed).

## Deployment (two laptops)
Full guide, including ZeroTier setup, firewall and the demo checklist: [DEPLOYMENT.md](DEPLOYMENT.md).
```bash
# Laptop 1 (model + API)
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Laptop 2 (evaluator UI) -> enter http://<laptop1-zerotier-ip>:8000 in the sidebar
streamlit run ui/app.py
```
Until the trained model and HMM pipeline are in `artifacts/` and `src/pathsnap/pipeline.py`, the API automatically falls back to a nearest-arc stub predictor (`api/stub.py`), and the startup log says so.

## Repository layout
| Path | Contents | Owner |
|---|---|---|
| `src/pathsnap/` | loader, projection, candidates, features, labels, metrics, HMM, pipeline, scoring | M1 / M2 / M3 |
| `scripts/` | dataset build, training, Optuna tuning, export, HMM evaluation, maps | all |
| `notebooks/` | EDA, model comparison, tuning, HMM analysis | all |
| `artifacts/` | trained model + configs used by the API | M2 / M3 |
| `results/` | CSVs, logs, plots, Folium maps (`m1/`, `m2/`, `m3/`, `api/`, `maps/`) | all |
| `api/` | FastAPI backend | M4 |
| `ui/` | evaluator UI | M4 |
| `report/` | report sections and figures | all |
| `data/` | raw / evaluator / processed data (git-ignored) | — |
