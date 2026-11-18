# PathSnap — GPS Map-Matching & Route Reconstruction

PathSnap takes noisy GPS trajectories and finds the road segment (directed arc) driven at every GPS observation. The pipeline is:
1. candidate generation;
2. an ML classifier that scores each (observation, candidate) pair (tuned LightGBM);
3. HMM/Viterbi decoding over road-network distances, which produces a connected route;
4. serving: a FastAPI backend on **Laptop 1** and a Streamlit evaluator UI on **Laptop 2**, connected over **ZeroTier**.

```
GPS track ─► candidates (≤50 m + 8-NN) ─► 23 features ─► LightGBM P(arc) ─► HMM / Viterbi ─► connected route
                                                                              (network distance transitions)
```

## Results (student_test, 10 trips, held out from training and tuning)

| Method | Top-1 (directed) | Top-1 (undirected) | Route F1 | Continuity |
|---|---|---|---|---|
| Nearest arc (geometry only) | 0.565 | 0.878 | 0.547 | 0.278 |
| Classifier only (tuned LightGBM) | 0.917 | 0.943 | 0.732 | 0.767 |
| **Classifier + HMM/Viterbi** | **0.937** | **0.957** | **0.973** | **0.905** |

Source: `results/m3/clf_vs_hmm_summary.csv`. All 9 models compared (cross-validation is GroupKFold by trip): `results/m2/model_comparison_v1.csv`. Bayesian tuning (Optuna TPE, 50 trials): `results/m2/optuna_trials_v1.csv` and `results/m2/tuned_vs_untuned_v1.csv`.

## Setup

```bash
git clone git@github.com:Mebin-11B/ML-Hack-PES.git && cd ML-Hack-PES
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                        # makes `import pathsnap` work everywhere
```
Unzip the organiser data package so that `data/raw/train/<trip_id>/` and `data/raw/student_test/<trip_id>/` exist. Raw data is **not** committed.

## Reproduce the experiments

```bash
# 1. Data exploration (CSVs + plots -> results/m1/) and Folium maps (-> results/maps/)
python scripts/run_eda.py --splits train student_test
python scripts/make_maps.py --split train

# 2. Candidate / feature tables (~10 s for all 80 train trips; labels from the route alignment in src/pathsnap/labels.py)
python scripts/build_dataset.py --split train
python scripts/build_dataset.py --split student_test

# 3. Compare 9 models (GroupKFold by trip) + plots
python scripts/train_models.py --tag v1 --obs-every 4
python scripts/plot_models.py --tag v1

# 4. Bayesian hyperparameter optimisation (Optuna TPE) + export the final model to artifacts/
python scripts/tune_optuna.py --model lgbm --trials 50 --timeout-min 40 --obs-every 4 --tag v1
python scripts/export_model.py --model lgbm --tag tuned --params-json results/m2/optuna_best_params_v1.json

# 5. HMM decoder tuning (train trips) and classifier-only vs HMM evaluation (student_test)
python scripts/tune_decoder.py
python scripts/eval_hmm.py --split student_test
```

## Deployment (two laptops)

The full guide is [DEPLOYMENT.md](DEPLOYMENT.md). It covers ZeroTier setup, firewall rules, the API reference and a demo checklist.

```bash
# Laptop 1 (model + API): evaluation trips go in data/evaluator_test/<trip_id>/
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Laptop 2 (evaluator UI): enter http://<laptop1-zerotier-ip>:8000 in the sidebar
streamlit run ui/app.py

# Rehearsal from Laptop 2: runs every trip through the API and records latency
python api/smoke_test.py --api http://<laptop1-zerotier-ip>:8000
```

**Leakage safeguards:**
- The predictor loads trips with `with_route=False`.
- Only `src/pathsnap/scoring.py` reads `.route`, and only after the prediction has been made, to compute the metrics.
- The API returns aggregate metrics only, never the reference route.
- Trips are discovered from the folder at runtime; nothing is hardcoded.

## Repository layout

| Path | Contents | Owner |
|---|---|---|
| `src/pathsnap/io.py`, `geo.py`, `candidates.py`, `features.py` | loader, per-trip UTM projection, STRtree candidate index, features | M1 |
| `src/pathsnap/models.py` | model zoo, GroupKFold CV harness | M2 |
| `src/pathsnap/labels.py`, `metrics.py`, `routing.py`, `hmm.py`, `pipeline.py`, `scoring.py` | route alignment labels, shared metrics, road graph and shortest paths, Viterbi, end-to-end predictor, server-side scoring | M3 |
| `scripts/` | data exploration, dataset build, training, tuning, export, HMM evaluation, maps | M1 / M2 / M3 |
| `notebooks/` | EDA, model comparison, tuning | M1 / M2 |
| `artifacts/` | `model.joblib` (tuned LightGBM), `feature_list.json`, `model_card.json`, `hmm_config.json` | M2 / M3 |
| `results/` | `m1/` EDA, `m2/` models and tuning, `m3/` HMM, `maps/` Folium HTML, `api/` request logs and latency | all |
| `api/` | FastAPI backend, stub predictor (testing), smoke test | M4 |
| `ui/` | Streamlit evaluator UI | M4 |
| `report/` | report sections, figures, `build_report.py` (sections → HTML → PDF), `screenshot_maps.py` | all / M4 |

## Report

```bash
python report/build_report.py    # -> report/PathSnap_Report.html (open in a browser, Print -> Save as PDF)
```

## AI-assistance disclosure

We used AI coding assistants to scaffold and debug code and to plan the work split. All experiments were run by the team. Every reported number comes from the code and the logged results in this repository.
