"""M2 — fit the chosen model on all valid train rows and export the artifact contract.

Writes (contract with M3/M4, team overview §6.3):
  artifacts/model.joblib       fitted estimator; predict_proba(df[feature_list])[:, 1] = P(true arc)
  artifacts/model_card.json    model name, params, feature-list hash, CV/ST scores, date, seed, versions
Also writes feature importance (boosting gain + permutation importance on student_test Top-1):
  results/m2/feature_importance_<tag>.csv  and  results/m2/plots/feature_importance_<tag>.png

Examples
--------
  # early export (best untuned model from the v0 sweep)
  python scripts/export_model.py --model lgbm --tag untuned --cv-csv results/m2/model_comparison_v0.csv

  # final export with Optuna's best params
  python scripts/export_model.py --model lgbm --tag tuned --params-json results/m2/optuna_best_params.json

  # harness test (writes to results/m2/toy/, never to artifacts/)
  python scripts/export_model.py --toy --model lgbm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.models import (MODEL_SPECS, RANDOM_STATE, TOY_FEATURES, _final_step,  # noqa: E402
                             eval_student_test, label_source, load_feature_list, make_toy_table, obs_metrics,
                             subsample_obs)


def lib_versions() -> dict:
    out = {"python": platform.python_version()}
    for lib in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "optuna", "joblib"):
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            pass
    return out


def gain_importance(est, features: list[str]) -> pd.Series | None:
    f = _final_step(est)
    try:
        import lightgbm as lgb
        if isinstance(f, lgb.LGBMClassifier):
            return pd.Series(f.booster_.feature_importance(importance_type="gain"), index=features)
    except ImportError:
        pass
    try:
        import xgboost as xgb
        if isinstance(f, xgb.XGBClassifier):
            g = f.get_booster().get_score(importance_type="gain")
            return pd.Series([g.get(c, 0.0) for c in features], index=features)
    except ImportError:
        pass
    if hasattr(f, "feature_importances_"):  # RF / ET (impurity-based)
        return pd.Series(f.feature_importances_, index=features)
    return None


def permutation_top1(est, df_valid: pd.DataFrame, features: list[str], n_repeats: int = 3) -> pd.DataFrame:
    """Drop in observation Top-1 when one feature column is shuffled (student_test, valid rows)."""
    rng = np.random.default_rng(RANDOM_STATE)
    base = obs_metrics(df_valid, est.predict_proba(df_valid[features])[:, 1])["top1"]
    rows = []
    for c in features:
        drops = []
        for _ in range(n_repeats):
            X = df_valid[features].copy()
            X[c] = rng.permutation(X[c].to_numpy())
            drops.append(base - obs_metrics(df_valid, est.predict_proba(X)[:, 1])["top1"])
        rows.append({"feature": c, "perm_top1_drop_mean": float(np.mean(drops)),
                     "perm_top1_drop_std": float(np.std(drops))})
    return pd.DataFrame(rows), base


def cv_row(cv_csv: Path | None, model: str) -> dict:
    if cv_csv is None or not cv_csv.exists():
        return {}
    df = pd.read_csv(cv_csv)
    r = df[df["model"] == model]
    if r.empty:
        return {}
    keep = ["cv_top1_mean", "cv_top1_std", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_logloss",
            "n_folds", "obs_every", "tag"]
    return {k: (None if pd.isna(v) else v) for k, v in r.iloc[0][keep].to_dict().items() if k in r}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--params-json", default=None, help="JSON with {'params': {...}} or a flat dict (e.g. Optuna best)")
    ap.add_argument("--tag", default="untuned", help="untuned | tuned (goes into the card + output names)")
    ap.add_argument("--cv-csv", default=None, help="comparison CSV to copy CV scores from (untuned export)")
    ap.add_argument("--train", default="data/processed/train_candidates.parquet")
    ap.add_argument("--test", default="data/processed/student_test_candidates.parquet")
    ap.add_argument("--features", default="artifacts/feature_list.json")
    ap.add_argument("--raw-test-dir", default="data/raw/student_test")
    ap.add_argument("--obs-every", type=int, default=1, help="1 = use every observation (final model)")
    ap.add_argument("--max-train-rows", default="spec", help="'spec' (model default cap), 'none', or an int")
    ap.add_argument("--no-importance", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="cuda = train XGBoost on the GPU (other models are CPU-only)")
    ap.add_argument("--toy", action="store_true")
    args = ap.parse_args()
    import pathsnap.models as _m
    _m.DEVICE = args.device

    spec = MODEL_SPECS[args.model]
    res_dir = ROOT / "results/m2" / ("toy" if args.toy else "")
    art_dir = res_dir / "artifacts" if args.toy else ROOT / "artifacts"
    for d in (art_dir, res_dir / "plots"):
        d.mkdir(parents=True, exist_ok=True)

    params, cv_info = {}, {}
    if args.params_json:
        obj = json.loads((ROOT / args.params_json).read_text())
        params = obj.get("params", obj)
        cv_info = {k: obj[k] for k in ("best_value", "objective", "cv_top1", "cv_logloss", "cv_macro_f1",
                                        "n_trials", "folds", "obs_every") if k in obj}
    cv_info.update(cv_row(ROOT / args.cv_csv if args.cv_csv else None, args.model))
    cap = {"spec": "spec", "none": None}.get(args.max_train_rows.lower(), None)
    if args.max_train_rows.lower() not in ("spec", "none"):
        cap = int(args.max_train_rows)

    if args.toy:
        df_tr, df_te = make_toy_table(15, seed=RANDOM_STATE), make_toy_table(4, seed=RANDOM_STATE + 1)
        features, raw_test = TOY_FEATURES, None
        labels = "toy"
    else:
        df_tr, df_te = pd.read_parquet(ROOT / args.train), pd.read_parquet(ROOT / args.test)
        features = load_feature_list(ROOT / args.features)
        raw_test = ROOT / args.raw_test_dir if (ROOT / args.raw_test_dir).exists() else None
        labels = f"train={label_source(ROOT, 'train')}; student_test={label_source(ROOT, 'student_test')}"
    for d in (df_tr, df_te):
        d["trip_id"] = d["trip_id"].astype(str)
        d["label_valid"] = d["label_valid"].astype(bool)
    df_tr = subsample_obs(df_tr, args.obs_every).reset_index(drop=True)

    print(f"fitting {args.model} ({args.tag}) on {df_tr.label_valid.sum():,} valid train rows ...")
    st = eval_student_test(args.model, df_tr, df_te, features, params=params,
                           raw_test_dir=raw_test, max_train_rows=cap)
    est = st["model"]

    # Laptop 1 may have no GPU: an XGBoost model trained on cuda is switched to CPU inference before saving
    f = _final_step(est)
    if hasattr(f, "get_params") and f.get_params().get("device", "cpu") != "cpu":
        f.set_params(device="cpu")
    joblib.dump(est, art_dir / "model.joblib", compress=3)
    full_params = {**spec.params, **params}
    st_scores = {k: st[k] for k in ("st_top1", "st_undirected_top1", "st_miss_rate", "st_macro_f1",
                                    "st_precision", "st_recall", "st_logloss", "route_continuity",
                                    "route_breaks")}
    card = {
        "model": f"{args.model} ({args.tag})",
        "model_key": args.model, "family": spec.family, "variant": args.tag,
        "params": full_params,
        "feature_list": features,
        "feature_list_sha256": hashlib.sha256(json.dumps(features).encode()).hexdigest(),
        "cv": cv_info,
        "student_test": {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in st_scores.items()},
        "st_confusion_thr0.5": st["st_confusion"],
        "n_train_rows": st["n_train_rows"], "obs_every": args.obs_every,
        "train_time_s": st["train_time_s"], "infer_ms_per_1k": st["infer_ms_per_1k"],
        "best_iteration": st["history"].get("best_iteration") if st["history"] else None,
        "train_date": datetime.now().isoformat(timespec="seconds"),
        "seed": RANDOM_STATE, "library_versions": lib_versions(),
        "toy": bool(args.toy),
        "train_device": args.device if args.model == "xgb" else "cpu",
        "label_source": labels,
        "notes": ("Labels are M1's fallback (nearest route arc) because M3's route alignment was not yet "
                  "available: candidate miss rate is 0 by construction and is not reported. Rebuild the "
                  "parquets and re-export once M3's labels.py is in.") if "fallback" in labels else "",
    }
    if "fallback" in labels:
        card["student_test"]["st_miss_rate"] = None
    (art_dir / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    print(f"wrote {art_dir / 'model.joblib'} and model_card.json | ST top1 {st['st_top1']:.4f} "
          f"logloss {st['st_logloss']:.4f}")

    # sanity: reload and check the contract
    reloaded = joblib.load(art_dir / "model.joblib")
    p = reloaded.predict_proba(df_te[features].head(1000))[:, 1]
    assert p.shape == (min(1000, len(df_te)),) and np.all((p >= 0) & (p <= 1))

    if not args.no_importance:
        dv = df_te[df_te["label_valid"]].reset_index(drop=True)
        perm, base = permutation_top1(est, dv, features)
        gain = gain_importance(est, features)
        if gain is not None:
            perm["gain_or_impurity"] = perm["feature"].map(gain)
        perm = perm.sort_values("perm_top1_drop_mean", ascending=False)
        perm.to_csv(res_dir / f"feature_importance_{args.tag}.csv", index=False)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = 2 if gain is not None else 1
        fig, axs = plt.subplots(1, n, figsize=(6 * n, 0.35 * len(features) + 1.5), squeeze=False)
        q = perm.iloc[::-1]
        axs[0, 0].barh(q.feature, q.perm_top1_drop_mean, xerr=q.perm_top1_drop_std, color="#4C72B0")
        axs[0, 0].set_title(f"Permutation importance (ST Top-1 drop, base {base:.3f})")
        if gain is not None:
            g = perm.sort_values("gain_or_impurity")
            axs[0, 1].barh(g.feature, g.gain_or_impurity, color="#DD8452")
            axs[0, 1].set_title("Gain (boosting) / impurity (trees)")
        fig.suptitle(f"{args.model} ({args.tag})")
        fig.tight_layout()
        fig.savefig(res_dir / "plots" / f"feature_importance_{args.tag}.png", dpi=150)
        print("feature importance ->", res_dir / f"feature_importance_{args.tag}.csv")


if __name__ == "__main__":
    main()
