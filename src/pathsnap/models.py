"""M2 - Model zoo: make_pipeline(name, params) for the >=5 compared models.

Binary task: for each (observation, candidate arc) row, P(candidate is the true arc).

Contents
--------
* ``MODEL_SPECS`` / ``make_pipeline(name, params)``  — the model zoo
* ``run_cv``             — GroupKFold-by-trip cross-validation
* ``eval_student_test``  — fit on all train rows, score the student_test split
* metric helpers (pair-level + observation-level), implemented to the FROZEN
  definitions in the team overview §6.6. Swap to ``pathsnap.metrics`` (M3)
  once it is pushed; numbers must match.
* ``make_toy_table``     — synthetic table with the §6.2 schema, for harness tests
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

RANDOM_STATE = 42
# "cuda" trains XGBoost on the GPU (only XGBoost has GPU support in our stack). Set via --device.
DEVICE = os.environ.get("PATHSNAP_DEVICE", "cpu")
KEYS = ["trip_id", "obs_idx"]
EPS = 1e-7


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    name: str
    family: str
    params: dict
    builder: Callable[[dict], object]
    max_train_rows: int | None = None   # cap (by observation) for slow learners
    early_stop: str | None = None       # "lgbm" | "xgb" -> inner trip-level val split
    available: bool = True

    def key_config(self, params: dict | None = None) -> str:
        p = params if params is not None else self.params
        return "; ".join(f"{k}={v}" for k, v in p.items())


def _scaled(est):
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


MODEL_SPECS: dict[str, ModelSpec] = {
    "logreg": ModelSpec(
        "logreg", "linear / probabilistic",
        dict(C=1.0, class_weight="balanced", max_iter=1000),
        lambda p: _scaled(LogisticRegression(random_state=RANDOM_STATE, **p)),
    ),
    "gnb": ModelSpec(
        "gnb", "probabilistic (naive Bayes)", dict(var_smoothing=1e-9),
        lambda p: GaussianNB(**p),
    ),
    "knn": ModelSpec(
        "knn", "instance-based",
        dict(n_neighbors=25, weights="distance"),
        lambda p: _scaled(KNeighborsClassifier(n_jobs=-1, **p)),
        max_train_rows=100_000,
    ),
    "rf": ModelSpec(
        "rf", "ensemble (bagging)",
        dict(n_estimators=300, max_depth=None, min_samples_leaf=5),
        lambda p: RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE, **p),
        max_train_rows=500_000,
    ),
    "et": ModelSpec(
        "et", "ensemble (randomised trees)",
        dict(n_estimators=300, max_depth=None, min_samples_leaf=5),
        lambda p: ExtraTreesClassifier(n_jobs=-1, random_state=RANDOM_STATE, **p),
        max_train_rows=500_000,
    ),
    "hgb": ModelSpec(
        "hgb", "boosting (sklearn hist)",
        dict(learning_rate=0.05, max_leaf_nodes=31, max_iter=500,
             early_stopping=True, validation_fraction=0.1, n_iter_no_change=30),
        lambda p: HistGradientBoostingClassifier(random_state=RANDOM_STATE, **p),
    ),
    "lgbm": ModelSpec(
        "lgbm", "boosting (LightGBM)",
        dict(n_estimators=500, learning_rate=0.05, num_leaves=31),
        lambda p: lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **p),
        early_stop="lgbm", available=lgb is not None,
    ),
    "xgb": ModelSpec(
        "xgb", "boosting (XGBoost)",
        dict(n_estimators=500, learning_rate=0.05, max_depth=6),
        lambda p: xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", device=DEVICE,
                                    eval_metric="logloss", early_stopping_rounds=30, **p),
        early_stop="xgb", available=xgb is not None,
    ),
    "mlp": ModelSpec(
        "mlp", "neural network",
        dict(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
             early_stopping=True, max_iter=200),
        lambda p: _scaled(MLPClassifier(random_state=RANDOM_STATE, **p)),
    ),
}


def make_pipeline(name: str, params: dict | None = None):
    """Build an unfitted estimator for ``name`` (defaults overridden by ``params``)."""
    spec = MODEL_SPECS[name]
    if not spec.available:
        raise ImportError(f"model '{name}' needs a library that is not installed")
    return spec.builder({**spec.params, **(params or {})})


def _final_step(est):
    return est[-1] if isinstance(est, Pipeline) else est


def fit_model(spec: ModelSpec, est, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
    """Fit ``est``. Boosting libs get an inner *trip-level* validation split for
    early stopping (so the outer validation fold is never used for stopping).
    Returns (fitted_est, history_dict)."""
    if spec.early_stop:
        inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
        tr, va = next(inner.split(X, y, groups))
        Xtr, ytr, Xva, yva = X.iloc[tr], y[tr], X.iloc[va], y[va]
        if spec.early_stop == "lgbm":
            kw = dict(eval_names=["train", "valid"], eval_metric="binary_logloss",
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
            try:  # LightGBM >= 4.7 API
                est.fit(Xtr, ytr, eval_X=(Xtr, Xva), eval_y=(ytr, yva), **kw)
            except TypeError:  # older LightGBM (team minimum is 4.1)
                est.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], **kw)
        else:
            est.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
    else:
        est.fit(X, y)
    return est, extract_history(est)


def extract_history(est) -> dict:
    """Real per-iteration histories only (no fabricated curves)."""
    f = _final_step(est)
    h: dict = {}
    if lgb is not None and isinstance(f, lgb.LGBMClassifier):
        r = f.evals_result_
        h = {"metric": "binary_logloss",
             "train": list(r["train"]["binary_logloss"]),
             "valid": list(r["valid"]["binary_logloss"]),
             "best_iteration": int(f.best_iteration_ or len(r["train"]["binary_logloss"]))}
    elif xgb is not None and isinstance(f, xgb.XGBClassifier):
        r = f.evals_result()
        h = {"metric": "logloss",
             "train": list(r["validation_0"]["logloss"]),
             "valid": list(r["validation_1"]["logloss"]),
             "best_iteration": int(getattr(f, "best_iteration", len(r["validation_0"]["logloss"]) - 1)) + 1}
    elif isinstance(f, HistGradientBoostingClassifier) and getattr(f, "do_early_stopping_", False):
        # sklearn stores scores = -loss; flip sign to get log loss
        h = {"metric": "log_loss", "train": [-float(s) for s in f.train_score_],
             "valid": [-float(s) for s in f.validation_score_], "best_iteration": int(f.n_iter_)}
    elif isinstance(f, MLPClassifier):
        h = {"metric": "train log loss / valid accuracy",
             "train": [float(v) for v in f.loss_curve_],
             "valid_accuracy": [float(v) for v in (f.validation_scores_ or [])],
             "best_iteration": int(f.n_iter_)}
    return h


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def load_feature_list(path: str | Path) -> list[str]:
    """Accepts list[str], list[{"name":..}], or {"features": [...]}."""
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, dict):
        obj = obj.get("features", obj.get("feature_list"))
    return [o["name"] if isinstance(o, dict) else str(o) for o in obj]


def label_source(root: str | Path, split: str) -> str:
    """Label provenance recorded by M1's build_dataset.py (results/m1/trip_summary_<split>.csv).
    'fallback_nearest_route_arc' = nearest route arc (M3 alignment not yet available): the true arc is
    always a candidate by construction, so candidate miss rate is NOT meaningful and must not be reported."""
    p = Path(root) / "results" / "m1" / f"trip_summary_{split}.csv"
    if not p.exists():
        return "unknown"
    vals = sorted(pd.read_csv(p, usecols=["label_source"])["label_source"].astype(str).unique())
    return "+".join(vals)


def subsample_obs(df: pd.DataFrame, every: int) -> pd.DataFrame:
    """Keep every k-th observation per trip, with ALL its candidates."""
    if every <= 1:
        return df
    return df[df["obs_idx"] % every == 0]


def cap_rows_by_obs(df: pd.DataFrame, max_rows: int | None, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Randomly drop whole observations until ~max_rows rows remain."""
    if max_rows is None or len(df) <= max_rows:
        return df
    keys = df[KEYS].drop_duplicates()
    frac = max_rows / len(df)
    keep = keys.sample(frac=frac, random_state=seed)
    return df.merge(keep, on=KEYS, how="inner")


# --------------------------------------------------------------------------- #
# Metrics (FROZEN definitions, team overview §6.6)
# --------------------------------------------------------------------------- #
try:  # M3's shared definitions (src/pathsnap/metrics.py) — used whenever available
    from pathsnap import metrics as team_metrics
    _ = (team_metrics.pair_metrics, team_metrics.top1_accuracy)
except (ImportError, AttributeError):  # pragma: no cover
    team_metrics = None


def pair_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    """Pair-level metrics at threshold 0.5. Delegates to M3's metrics.pair_metrics (verified identical,
    except log-loss clipping: team uses 1e-9). Confusion matrix always 2x2 ([[TN, FP], [FN, TP]])."""
    if team_metrics is not None and thr == 0.5:
        r = team_metrics.pair_metrics(y, p)
        return {"accuracy": r["acc"], "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                "macro_f1": r["macro_f1"], "logloss": r["logloss"],
                "confusion": confusion_matrix(y, (np.asarray(p) >= thr).astype(int), labels=[0, 1]).tolist()}
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "logloss": log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def argmax_per_obs(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """One row per (trip_id, obs_idx): the candidate with max probability."""
    d = df[KEYS + ["arc_id", "gt_arc"]].copy()
    d["p"] = np.asarray(p)
    d = d.sort_values(KEYS + ["p"], ascending=[True, True, False], kind="stable")
    return d.drop_duplicates(KEYS, keep="first").reset_index(drop=True)


def obs_metrics(df: pd.DataFrame, p: np.ndarray, twin_maps: dict | None = None) -> dict:
    """Observation Top-1 (primary), undirected Top-1, candidate miss rate.
    ``df`` must contain only label_valid rows."""
    best = argmax_per_obs(df, p)
    correct = best["arc_id"].to_numpy() == best["gt_arc"].to_numpy()
    top1 = (team_metrics.top1_accuracy(best["arc_id"], best["gt_arc"], np.ones(len(best), bool))
            if team_metrics is not None else float(correct.mean()))
    out = {"top1": float(top1), "n_obs": int(len(best))}  # pooled over all valid observations
    out["miss_rate"] = float(1.0 - df.groupby(KEYS)["label"].max().mean())
    if twin_maps:
        und = correct.copy()
        for tid, g in best.groupby("trip_id"):
            tw = twin_maps.get(tid)
            if tw is None:
                continue
            a = g["arc_id"].to_numpy()
            und[g.index.to_numpy()] |= tw[a] == g["gt_arc"].to_numpy()
        out["undirected_top1"] = float(und.mean())
    return out


def route_continuity(best: pd.DataFrame, arcs_by_trip: dict) -> dict:
    """Collapse consecutive duplicate arcs; share of consecutive pairs (a→b) with
    to_node(a) == from_node(b). ``best`` = argmax_per_obs output (all obs)."""
    conn = pairs = 0
    for tid, g in best.sort_values(KEYS).groupby("trip_id"):
        arcs = arcs_by_trip.get(tid)
        if arcs is None:
            continue
        seq = g["arc_id"].to_numpy()
        seq = seq[seq >= 0]
        if len(seq) < 2:
            continue
        seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
        ok = arcs[seq[:-1], 1] == arcs[seq[1:], 0]
        conn += int(ok.sum())
        pairs += len(ok)
    return {"route_continuity": conn / pairs if pairs else float("nan"),
            "route_breaks": pairs - conn}


def read_arcs(raw_split_dir: str | Path, trip_id: str) -> np.ndarray | None:
    path = Path(raw_split_dir) / trip_id / f"{trip_id}.arcs"
    if not path.exists():
        return None
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy(dtype=np.int64)


def twin_lookup(arcs: np.ndarray) -> np.ndarray:
    """twin[a] = arc id of the reverse arc (b→a), or -1."""
    n = int(arcs.max()) + 1
    key = arcs[:, 0] * n + arcs[:, 1]
    rev = arcs[:, 1] * n + arcs[:, 0]
    s = pd.Series(np.arange(len(arcs)), index=key)
    s = s[~s.index.duplicated()]
    return s.reindex(rev).fillna(-1).to_numpy(dtype=np.int64)


def load_graph_info(raw_split_dir: str | Path | None, trip_ids) -> tuple[dict, dict]:
    arcs_by_trip, twins = {}, {}
    if raw_split_dir is None:
        return arcs_by_trip, twins
    for tid in trip_ids:
        a = read_arcs(raw_split_dir, str(tid))
        if a is not None:
            arcs_by_trip[tid] = a
            twins[tid] = twin_lookup(a)
    return arcs_by_trip, twins


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def _predict(est, X) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    p = est.predict_proba(X)[:, 1]
    ms_per_1k = (time.perf_counter() - t0) * 1000 / max(len(X), 1) * 1000
    return p, ms_per_1k


def run_cv(name: str, df: pd.DataFrame, features: list[str], n_splits: int = 5,
           params: dict | None = None, max_train_rows: int | None | str = "spec") -> tuple[dict, list[dict]]:
    """GroupKFold by trip_id on label_valid rows. Returns (summary, per_fold_rows)."""
    spec = MODEL_SPECS[name]
    cap = spec.max_train_rows if max_train_rows == "spec" else max_train_rows
    df = df[df["label_valid"]].reset_index(drop=True)
    groups = df["trip_id"].to_numpy()
    folds = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(df, groups=groups)):
        dtr = cap_rows_by_obs(df.iloc[tr], cap, seed=RANDOM_STATE + k).reset_index(drop=True)
        dva = df.iloc[va].reset_index(drop=True)
        est = make_pipeline(name, params)
        t0 = time.perf_counter()
        est, _ = fit_model(spec, est, dtr[features], dtr["label"].to_numpy(), dtr["trip_id"].to_numpy())
        fit_s = time.perf_counter() - t0
        p_va, ms1k = _predict(est, dva[features])
        p_tr, _ = _predict(est, dtr[features])
        pm = pair_metrics(dva["label"].to_numpy(), p_va)
        om = obs_metrics(dva, p_va)
        folds.append({
            "model": name, "fold": k, "n_train_rows": len(dtr), "n_val_rows": len(dva),
            "n_val_trips": dva["trip_id"].nunique(),
            "train_top1": obs_metrics(dtr, p_tr)["top1"],
            "val_top1": om["top1"], "val_miss_rate": om["miss_rate"],
            **{f"val_{m}": pm[m] for m in ("accuracy", "precision", "recall", "f1", "macro_f1", "logloss")},
            "fit_time_s": fit_s, "infer_ms_per_1k": ms1k,
        })
    f = pd.DataFrame(folds)
    summary = {
        "cv_macro_f1_mean": f["val_macro_f1"].mean(), "cv_macro_f1_std": f["val_macro_f1"].std(ddof=0),
        "cv_top1_mean": f["val_top1"].mean(), "cv_top1_std": f["val_top1"].std(ddof=0),
        "cv_logloss": f["val_logloss"].mean(), "cv_train_top1_mean": f["train_top1"].mean(),
        "cv_fit_time_s_mean": f["fit_time_s"].mean(),
    }
    return summary, folds


def eval_student_test(name: str, df_train: pd.DataFrame, df_test: pd.DataFrame, features: list[str],
                      params: dict | None = None, raw_test_dir: str | Path | None = None,
                      max_train_rows: int | None | str = "spec") -> dict:
    """Fit on all valid train rows (capped per spec), score student_test.
    Top-1 / pair metrics on valid rows; route continuity on ALL test observations."""
    spec = MODEL_SPECS[name]
    cap = spec.max_train_rows if max_train_rows == "spec" else max_train_rows
    dtr = cap_rows_by_obs(df_train[df_train["label_valid"]], cap).reset_index(drop=True)
    est = make_pipeline(name, params)
    t0 = time.perf_counter()
    est, hist = fit_model(spec, est, dtr[features], dtr["label"].to_numpy(), dtr["trip_id"].to_numpy())
    train_time = time.perf_counter() - t0

    dte = df_test.reset_index(drop=True)
    p_all, ms1k = _predict(est, dte[features])
    valid = dte["label_valid"].to_numpy()
    dv = dte[valid].reset_index(drop=True)
    pv = p_all[valid]

    arcs_by_trip, twins = load_graph_info(raw_test_dir, dte["trip_id"].unique())
    pm = pair_metrics(dv["label"].to_numpy(), pv)
    om = obs_metrics(dv, pv, twins)
    rc = route_continuity(argmax_per_obs(dte, p_all), arcs_by_trip) if arcs_by_trip else \
        {"route_continuity": float("nan"), "route_breaks": np.nan}
    return {
        "model": est, "history": hist, "p_test": p_all,
        "st_top1": om["top1"], "st_undirected_top1": om.get("undirected_top1", np.nan),
        "st_miss_rate": om["miss_rate"], "st_macro_f1": pm["macro_f1"],
        "st_precision": pm["precision"], "st_recall": pm["recall"], "st_logloss": pm["logloss"],
        "st_confusion": pm["confusion"], **rc,
        "train_time_s": train_time, "infer_ms_per_1k": ms1k, "n_train_rows": len(dtr),
    }


# --------------------------------------------------------------------------- #
# Synthetic table (harness testing only — never reported)
# --------------------------------------------------------------------------- #
TOY_FEATURES = ["dist_m", "dist_rank", "rel_dist", "heading_cos", "heading_sin", "speed_mps",
                "arc_len_m", "pos_along", "n_cands", "is_twin_pair", "win_dist_2"]


def make_toy_table(n_trips: int = 12, n_obs: int = 250, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_trips):
        tid = f"{t:08d}"
        for o in range(n_obs):
            k = int(rng.integers(4, 20))
            gt = int(rng.integers(0, 1000)) * 2
            arcs = np.r_[gt, gt + 1, rng.choice(np.arange(2001, 5000), k - 2, replace=False)]
            dist = np.r_[abs(rng.normal(0, 7)), 0, rng.uniform(3, 60, k - 2)]
            dist[1] = dist[0]  # twin: same geometry, opposite direction
            hcos = np.r_[rng.normal(0.9, 0.15), rng.normal(-0.9, 0.15), rng.uniform(-1, 1, k - 2)]
            order = np.argsort(dist).argsort()
            rows.append(pd.DataFrame({
                "trip_id": tid, "obs_idx": o, "t": float(o), "arc_id": arcs, "dist_m": dist,
                "label": (arcs == gt).astype(int), "label_valid": bool(rng.random() > 0.03),
                "gt_arc": gt, "dist_rank": order, "rel_dist": dist - dist.min(),
                "heading_cos": np.clip(hcos, -1, 1), "heading_sin": rng.normal(0, 0.3, k),
                "speed_mps": abs(rng.normal(10, 4)), "arc_len_m": rng.uniform(10, 300, k),
                "pos_along": rng.uniform(0, 1, k), "n_cands": k,
                "is_twin_pair": np.r_[1, 1, rng.integers(0, 2, k - 2)],
                "win_dist_2": dist + rng.normal(0, 3, k),
            }))
    return pd.concat(rows, ignore_index=True)
