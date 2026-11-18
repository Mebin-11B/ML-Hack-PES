"""M2 — train & compare the model zoo (GroupKFold by trip) + student_test evaluation.

Examples
--------
  # harness smoke test on synthetic data (writes to results/m2/toy/, never reported)
  python scripts/train_models.py --toy --models logreg,gnb,rf,hgb --folds 3

  # v0 sweep on M1's parquet, every 4th observation per trip
  python scripts/train_models.py --tag v0 --obs-every 4

  # full comparison on v1 features
  python scripts/train_models.py --obs-every 2
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.models import (MODEL_SPECS, RANDOM_STATE, TOY_FEATURES, eval_student_test,  # noqa: E402
                             label_source, load_feature_list, make_toy_table, run_cv,
                             subsample_obs)

COLUMNS = ["model", "family", "key_config", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean",
           "cv_top1_std", "cv_logloss", "st_top1", "st_macro_f1", "st_precision", "st_recall",
           "st_logloss", "route_continuity", "train_time_s", "infer_ms_per_1k", "n_train_rows", "seed",
           # extras (kept after the report-table columns)
           "cv_train_top1_mean", "st_undirected_top1", "st_miss_rate", "route_breaks", "n_folds",
           "obs_every", "tag", "label_source", "device"]


def upsert_csv(path: Path, rows: pd.DataFrame, key: list[str]):
    if path.exists():
        old = pd.read_csv(path, dtype={"model": str})
        old = old.merge(rows[key], on=key, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        rows = pd.concat([old, rows], ignore_index=True)
    rows.to_csv(path, index=False)


def save_preds(path: Path, df: pd.DataFrame, p: np.ndarray):
    out = df[["trip_id", "obs_idx", "arc_id", "gt_arc", "label", "label_valid"]].copy()
    out["p"] = p.astype(np.float32)
    try:
        out.to_parquet(path.with_suffix(".parquet"), index=False)
    except ImportError:
        out.to_csv(path.with_suffix(".csv.gz"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/processed/train_candidates.parquet")
    ap.add_argument("--test", default="data/processed/student_test_candidates.parquet")
    ap.add_argument("--features", default="artifacts/feature_list.json")
    ap.add_argument("--raw-test-dir", default="data/raw/student_test",
                    help="for undirected Top-1 + route continuity (reads .arcs only)")
    ap.add_argument("--models", default="all", help="comma list or 'all': " + ",".join(MODEL_SPECS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--obs-every", type=int, default=1, help="keep every k-th observation per trip (train)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. v0")
    ap.add_argument("--out", default="results/m2")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="cuda = train XGBoost on the GPU (other models are CPU-only)")
    ap.add_argument("--toy", action="store_true", help="synthetic data smoke test")
    args = ap.parse_args()
    import pathsnap.models as _m
    _m.DEVICE = args.device

    out = ROOT / args.out / ("toy" if args.toy else "")
    for sub in ("", "configs", "histories", "preds", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out / "logs" / f"train_models{sfx}.log", encoding="utf-8")])
    log = logging.getLogger("m2")

    if args.toy:
        df_tr = make_toy_table(n_trips=15, seed=RANDOM_STATE)
        df_te = make_toy_table(n_trips=4, seed=RANDOM_STATE + 1)
        features, raw_test = TOY_FEATURES, None
        labels = "toy"
    else:
        df_tr = pd.read_parquet(ROOT / args.train)
        df_te = pd.read_parquet(ROOT / args.test)
        features = load_feature_list(ROOT / args.features)
        raw_test = ROOT / args.raw_test_dir if (ROOT / args.raw_test_dir).exists() else None
        labels = f"train={label_source(ROOT, 'train')}; student_test={label_source(ROOT, 'student_test')}"
    fallback = "fallback" in labels
    for d in (df_tr, df_te):
        d["trip_id"] = d["trip_id"].astype(str)
        d["label_valid"] = d["label_valid"].astype(bool)
    df_tr = subsample_obs(df_tr, args.obs_every).reset_index(drop=True)
    missing = [f for f in features if f not in df_tr.columns]
    if missing:
        raise SystemExit(f"features missing from parquet: {missing}")
    log.info("labels: %s%s", labels,
             " -> candidate miss rate is 0 by construction and is NOT reported" if fallback else "")
    log.info("train rows=%d trips=%d | test rows=%d trips=%d | %d features | obs_every=%d",
             len(df_tr), df_tr.trip_id.nunique(), len(df_te), df_te.trip_id.nunique(),
             len(features), args.obs_every)

    names = list(MODEL_SPECS) if args.models == "all" else args.models.split(",")
    for name in names:
        spec = MODEL_SPECS[name]
        if not spec.available:
            log.warning("skip %s: library not installed", name)
            continue
        log.info("=== %s (%s) | %s", name, spec.family, spec.key_config())
        t0 = time.perf_counter()
        try:
            cv, folds = run_cv(name, df_tr, features, n_splits=args.folds)
            st = eval_student_test(name, df_tr, df_te, features, raw_test_dir=raw_test)
        except Exception:
            log.exception("model %s failed", name)
            continue
        row = {"model": name, "family": spec.family, "key_config": spec.key_config(), **cv,
               **{k: v for k, v in st.items() if k not in ("model", "history", "p_test", "st_confusion")},
               "seed": RANDOM_STATE, "n_folds": args.folds, "obs_every": args.obs_every, "tag": args.tag,
               "label_source": labels, "device": args.device if name == "xgb" else "cpu"}
        if fallback:
            row["st_miss_rate"] = float("nan")
            for f in folds:
                f["val_miss_rate"] = float("nan")
        upsert_csv(out / f"model_comparison{sfx}.csv", pd.DataFrame([row])[COLUMNS], ["model"])
        upsert_csv(out / f"cv_folds{sfx}.csv", pd.DataFrame(folds), ["model", "fold"])
        (out / "configs" / f"{name}{sfx}.json").write_text(json.dumps({
            "model": name, "family": spec.family, "params": spec.params,
            "max_train_rows": spec.max_train_rows, "early_stop": spec.early_stop, "seed": RANDOM_STATE,
            "features": features, "folds": args.folds, "obs_every": args.obs_every,
            "st_confusion": st["st_confusion"], "label_source": labels, "python": platform.python_version()}, indent=2, default=str))
        if st["history"]:
            (out / "histories" / f"{name}{sfx}.json").write_text(json.dumps(st["history"]))
        save_preds(out / "preds" / f"st_{name}{sfx}", df_te, st["p_test"])
        log.info("%s done in %.1fs | CV top1 %.4f±%.4f macroF1 %.4f logloss %.4f | ST top1 %.4f macroF1 %.4f",
                 name, time.perf_counter() - t0, cv["cv_top1_mean"], cv["cv_top1_std"],
                 cv["cv_macro_f1_mean"], cv["cv_logloss"], st["st_top1"], st["st_macro_f1"])

    log.info("wrote %s", out / f"model_comparison{sfx}.csv")


if __name__ == "__main__":
    main()
