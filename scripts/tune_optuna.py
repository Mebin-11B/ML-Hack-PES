"""M2 — Bayesian hyperparameter optimisation (Optuna TPE) + tuned-vs-untuned comparison.

Objective = mean GroupKFold (by trip) observation Top-1 on the TRAIN trips only
(or log loss with --objective logloss). student_test is used only AFTER the study,
for the tuned-vs-untuned table. Evaluator data is never touched.

Outputs (results/m2/):
  optuna_trials<sfx>.csv          study.trials_dataframe() + per-trial CV metrics
  optuna_best_params<sfx>.json    best params, best value, settings, wall time
  tuned_vs_untuned<sfx>.csv       same folds / rows / metrics for both
  plots/optuna_history<sfx>.png, plots/optuna_param_importance<sfx>.png

Examples
--------
  python scripts/tune_optuna.py --model lgbm --trials 50 --timeout-min 40 --obs-every 4
  python scripts/tune_optuna.py --toy --model lgbm --trials 8 --folds 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.models import (MODEL_SPECS, RANDOM_STATE, TOY_FEATURES, cap_rows_by_obs,  # noqa: E402
                             eval_student_test, fit_model, label_source, load_feature_list, make_pipeline,
                             make_toy_table, obs_metrics, pair_metrics, run_cv, subsample_obs)


# Search spaces — each range is justified in report/sections/04_tuning.md
def space(trial: optuna.Trial, model: str) -> dict:
    if model == "lgbm":
        return dict(
            n_estimators=2000,  # upper bound; early stopping on an inner trip split picks the count
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),   # feature_fraction
            subsample=trial.suggest_float("subsample", 0.5, 1.0),                 # bagging_fraction
            subsample_freq=1,
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),     # lambda_l1
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),   # lambda_l2
        )
    if model == "xgb":
        return dict(
            n_estimators=2000,
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if model == "hgb":
        return dict(
            max_iter=1000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        )
    if model in ("rf", "et"):
        depth = trial.suggest_categorical("max_depth", [None, 8, 12, 16, 24, 32])
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=depth,
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        )
    if model == "mlp":
        width = trial.suggest_categorical("width", [32, 64, 128])
        depth = trial.suggest_int("depth", 1, 3)
        return dict(
            hidden_layer_sizes=tuple([width] * depth),
            alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        )
    if model == "logreg":
        return dict(C=trial.suggest_float("C", 1e-3, 100.0, log=True),
                    class_weight=trial.suggest_categorical("class_weight", ["balanced", None]))
    raise SystemExit(f"no search space defined for {model}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lgbm", choices=list(MODEL_SPECS))
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--timeout-min", type=float, default=40.0)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--obs-every", type=int, default=4, help="subsample observations for speed")
    ap.add_argument("--objective", choices=["top1", "logloss"], default="top1")
    ap.add_argument("--train", default="data/processed/train_candidates.parquet")
    ap.add_argument("--test", default="data/processed/student_test_candidates.parquet")
    ap.add_argument("--features", default="artifacts/feature_list.json")
    ap.add_argument("--raw-test-dir", default="data/raw/student_test")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-compare", action="store_true", help="skip tuned-vs-untuned table")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="cuda = train XGBoost on the GPU (other models are CPU-only)")
    ap.add_argument("--toy", action="store_true")
    args = ap.parse_args()
    import pathsnap.models as _m
    _m.DEVICE = args.device

    out = ROOT / "results/m2" / ("toy" if args.toy else "")
    for sub in ("", "plots", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out / "logs" / f"tune_optuna{sfx}.log", encoding="utf-8")])
    log = logging.getLogger("m2.optuna")

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
    df_tr = subsample_obs(df_tr, args.obs_every)
    df_tr = df_tr[df_tr["label_valid"]].reset_index(drop=True)
    spec = MODEL_SPECS[args.model]
    folds = list(GroupKFold(n_splits=args.folds).split(df_tr, groups=df_tr["trip_id"]))
    log.info("tuning %s | %d valid train rows, %d trips | %d folds | objective=%s | %d trials / %.0f min",
             args.model, len(df_tr), df_tr.trip_id.nunique(), args.folds, args.objective,
             args.trials, args.timeout_min)

    def objective(trial: optuna.Trial) -> float:
        params = space(trial, args.model)
        top1s, lls, f1s, fit_s = [], [], [], 0.0
        for k, (tr, va) in enumerate(folds):
            dtr = cap_rows_by_obs(df_tr.iloc[tr], spec.max_train_rows, seed=RANDOM_STATE + k).reset_index(drop=True)
            dva = df_tr.iloc[va].reset_index(drop=True)
            est = make_pipeline(args.model, params)
            t0 = time.perf_counter()
            est, _ = fit_model(spec, est, dtr[features], dtr["label"].to_numpy(), dtr["trip_id"].to_numpy())
            fit_s += time.perf_counter() - t0
            p = est.predict_proba(dva[features])[:, 1]
            top1s.append(obs_metrics(dva, p)["top1"])
            pm = pair_metrics(dva["label"].to_numpy(), p)
            lls.append(pm["logloss"]); f1s.append(pm["macro_f1"])
            running = np.mean(top1s) if args.objective == "top1" else np.mean(lls)
            trial.report(running, step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_top1", float(np.mean(top1s)))
        trial.set_user_attr("cv_top1_std", float(np.std(top1s)))
        trial.set_user_attr("cv_logloss", float(np.mean(lls)))
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("fit_time_s", fit_s)
        return float(np.mean(top1s) if args.objective == "top1" else np.mean(lls))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize" if args.objective == "top1" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name=f"{args.model}{sfx}")
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout_min * 60,
                   callbacks=[lambda s, t: log.info("trial %d %s value=%s best=%s", t.number, t.state.name,
                                                    t.value, s.best_value if s.best_trial else None)])
    wall = time.perf_counter() - t0

    study.trials_dataframe().to_csv(out / f"optuna_trials{sfx}.csv", index=False)
    best = study.best_trial
    best_params = space(optuna.trial.FixedTrial(best.params), args.model)  # full dict incl. fixed values
    best_json = {
        "model": args.model, "params": best_params, "raw_trial_params": best.params,
        "best_value": best.value, "objective": f"mean GroupKFold obs {args.objective}",
        "cv_top1": best.user_attrs.get("cv_top1"), "cv_logloss": best.user_attrs.get("cv_logloss"),
        "cv_macro_f1": best.user_attrs.get("cv_macro_f1"), "best_trial": best.number,
        "n_trials": len(study.trials),
        "n_complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "folds": args.folds, "obs_every": args.obs_every, "sampler": "TPESampler(seed=42)",
        "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=1)", "wall_time_s": wall,
        "toy": bool(args.toy), "label_source": labels, "device": args.device,
    }
    (out / f"optuna_best_params{sfx}.json").write_text(json.dumps(best_json, indent=2, default=str))
    log.info("best trial %d value %.4f in %.0fs: %s", best.number, best.value, wall, best.params)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # optuna marks matplotlib plots as experimental
        ax = plot_optimization_history(study)
        ax.figure.set_size_inches(8, 4); ax.figure.tight_layout()
        ax.figure.savefig(out / "plots" / f"optuna_history{sfx}.png", dpi=150); plt.close(ax.figure)
        try:
            ax = plot_param_importances(study)
            ax.figure.set_size_inches(7, 4); ax.figure.tight_layout()
            ax.figure.savefig(out / "plots" / f"optuna_param_importance{sfx}.png", dpi=150); plt.close(ax.figure)
        except Exception as e:  # needs >=2 complete trials
            log.warning("param importance plot skipped: %s", e)

    if args.skip_compare:
        return
    # Tuned vs untuned: identical rows, folds, metrics
    rows = []
    for variant, params in (("untuned", None), ("tuned", best_params)):
        cv, _ = run_cv(args.model, df_tr, features, n_splits=args.folds, params=params)
        st = eval_student_test(args.model, df_tr, df_te, features, params=params, raw_test_dir=raw_test)
        rows.append({"model": args.model, "variant": variant,
                     "cv_top1": cv["cv_top1_mean"], "cv_top1_std": cv["cv_top1_std"],
                     "cv_macro_f1": cv["cv_macro_f1_mean"], "cv_logloss": cv["cv_logloss"],
                     "st_top1": st["st_top1"], "st_undirected_top1": st["st_undirected_top1"],
                     "st_macro_f1": st["st_macro_f1"], "st_logloss": st["st_logloss"],
                     "route_continuity": st["route_continuity"], "train_time_s": st["train_time_s"],
                     "infer_ms_per_1k": st["infer_ms_per_1k"], "folds": args.folds,
                     "obs_every": args.obs_every, "label_source": labels})
        log.info("%s: CV top1 %.4f logloss %.4f | ST top1 %.4f logloss %.4f", variant,
                 cv["cv_top1_mean"], cv["cv_logloss"], st["st_top1"], st["st_logloss"])
    pd.DataFrame(rows).to_csv(out / f"tuned_vs_untuned{sfx}.csv", index=False)
    log.info("wrote %s", out / f"tuned_vs_untuned{sfx}.csv")


if __name__ == "__main__":
    main()
