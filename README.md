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

## Models — candidate classifier (M2)

Each row of the candidate table is one *(GPS observation, candidate arc)* pair. The classifier outputs
`P(candidate is the true arc)`; per observation, the highest-probability candidate is the classifier-only
prediction, and the probabilities are the HMM emission scores (so calibration matters, not only accuracy).

### Model zoo (`src/pathsnap/models.py`)
| Key | Model | Family | Starting configuration | Notes |
|---|---|---|---|---|
| `logreg` | Logistic Regression | linear / probabilistic | StandardScaler, C=1.0, class_weight=balanced, max_iter=1000 | baseline |
| `gnb` | Gaussian Naive Bayes | probabilistic | var_smoothing=1e-9 | feature-independence assumption is violated (correlated distance features) |
| `knn` | k-Nearest Neighbours | instance-based | StandardScaler, k=25, distance-weighted | trained on ≤100k rows (slow inference) |
| `rf` | Random Forest | bagging ensemble | 300 trees, max_depth=None, min_samples_leaf=5 | trained on ≤500k rows |
| `et` | Extra Trees | randomised-tree ensemble | 300 trees, max_depth=None, min_samples_leaf=5 | trained on ≤500k rows |
| `hgb` | HistGradientBoosting | boosting (sklearn) | lr=0.05, max_leaf_nodes=31, max_iter=500, early stopping | |
| `lgbm` | LightGBM | boosting | lr=0.05, num_leaves=31, n_estimators=500, early stopping (30 rounds) | |
| `xgb` | XGBoost | boosting | eta=0.05, max_depth=6, 500 rounds, hist, early stopping (30 rounds) | |
| `mlp` | MLP | neural network | StandardScaler, (64, 32), ReLU, Adam, early_stopping=True | |

All models use `random_state=42`. Scalers sit inside sklearn `Pipeline`s, so they are fit on training folds only.
LightGBM/XGBoost early-stop on an inner **trip-level** split of the training fold, never on the validation fold.
Row caps drop whole observations (all candidates of a GPS point are kept together).

### Evaluation protocol
- **Cross-validation:** `GroupKFold` by `trip_id` on the 80 train trips (points of one trip are highly
  correlated, so a random split would leak). Only `label_valid == True` rows are used for training and metrics.
- **Development test:** the 10 `student_test` trips. Evaluator trips are never used for training or tuning.
- **Primary metric:** observation **Top-1 accuracy** (argmax candidate per GPS point == aligned ground-truth arc).
  Pair-level accuracy is misleading here because ~1 in 10–30 candidates is positive.
- **Also reported:** macro-F1 / precision / recall / log loss at threshold 0.5, undirected Top-1 (reverse twin
  arc counts as correct), candidate miss rate, route continuity of the argmax sequence, train time,
  inference time (ms per 1k rows), train-fold Top-1 (over-fitting check).
- **Selection rule:** highest CV Top-1; ties broken by log loss (probabilities feed the HMM), then inference speed.

### Hyperparameter tuning (`scripts/tune_optuna.py`)
Optuna **TPE** sampler (seed 42) with a median pruner. The objective is mean GroupKFold observation Top-1 on the
train trips only (`--objective logloss` is also available). LightGBM search space: `num_leaves` 15–255 (log),
`learning_rate` 0.01–0.2 (log), `min_child_samples` 10–200 (log), `colsample_bytree` 0.5–1, `subsample` 0.5–1,
`reg_alpha` / `reg_lambda` 1e-3–10 (log), up to 2000 trees with early stopping. Spaces for XGBoost,
HistGB, RF/ET, MLP and LogReg are defined in the same script. The tuned and untuned models are then compared on
identical folds, rows and metrics.

### Reproduce
```bash
# 1. compare all models (needs data/processed/*_candidates.parquet + artifacts/feature_list.json from M1)
python scripts/train_models.py --tag v0 --obs-every 4      # -> results/m2/model_comparison_v0.csv
python scripts/plot_models.py  --tag v0                    # -> results/m2/plots/
# 2. early export of the best untuned model
python scripts/export_model.py --model lgbm --tag untuned --cv-csv results/m2/model_comparison_v0.csv
# 3. Bayesian tuning + tuned-vs-untuned table
python scripts/tune_optuna.py --model lgbm --trials 50 --timeout-min 40 --obs-every 4
# 4. final export with the tuned parameters
python scripts/export_model.py --model lgbm --tag tuned --params-json results/m2/optuna_best_params.json
```
Every script accepts `--toy` to run on a synthetic table (output in `results/m2/toy/`, git-ignored, never reported).
`lgbm` above is a placeholder; the selected model is whichever wins step 1.

### Outputs
| File | Contents |
|---|---|
| `results/m2/model_comparison*.csv` | one row per model: CV and student_test metrics, timings, seed |
| `results/m2/cv_folds*.csv`, `configs/`, `histories/`, `logs/` | per-fold scores, configs, real training histories, run logs |
| `results/m2/optuna_trials*.csv`, `optuna_best_params*.json` | every Optuna trial; best parameters + settings + wall time |
| `results/m2/tuned_vs_untuned*.csv` | tuned vs untuned on the same splits |
| `results/m2/feature_importance_*.csv` | permutation importance (student_test Top-1 drop) + gain |
| `artifacts/model.joblib` | fitted estimator: `predict_proba(df[feature_list])[:, 1]` |
| `artifacts/model_card.json` | model, params, feature-list hash, CV / student_test scores, date, seed, library versions |

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
