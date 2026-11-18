"""M2 — full metric table per model from saved outputs (no retraining).

CV metrics  = mean ± std over folds from results/m2/cv_folds<sfx>.csv
ST metrics  = recomputed from saved student_test predictions results/m2/preds/st_<model><sfx>.parquet
              (label_valid rows; pair metrics at threshold 0.5) + ROC-AUC / PR-AUC (threshold-free).

  python scripts/summarize_metrics.py --tag v0   ->  results/m2/metrics_summary_v0.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.models import obs_metrics, pair_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--toy", action="store_true")
    args = ap.parse_args()
    base = ROOT / "results/m2" / ("toy" if args.toy else "")
    sfx = f"_{args.tag}" if args.tag else ""

    comp = pd.read_csv(base / f"model_comparison{sfx}.csv")
    folds = pd.read_csv(base / f"cv_folds{sfx}.csv")
    rows = []
    for _, c in comp.iterrows():
        m = c["model"]
        f = folds[folds["model"] == m]
        r = {"model": m}
        for k in ("accuracy", "precision", "recall", "f1", "macro_f1", "logloss", "top1"):
            r[f"cv_{k}_mean"] = f[f"val_{k}"].mean()
            r[f"cv_{k}_std"] = f[f"val_{k}"].std(ddof=0)
        r["cv_train_top1_mean"] = f["train_top1"].mean()

        pp = base / "preds" / f"st_{m}{sfx}.parquet"
        if pp.exists():
            p = pd.read_parquet(pp)
            p = p[p["label_valid"].astype(bool)].reset_index(drop=True)
            y, pr = p["label"].to_numpy(), p["p"].to_numpy().astype(float)
            pm = pair_metrics(y, pr)
            (tn, fp), (fn, tp) = pm["confusion"]
            r.update({f"st_{k}": pm[k] for k in ("accuracy", "precision", "recall", "f1", "macro_f1", "logloss")})
            r.update(st_roc_auc=roc_auc_score(y, pr), st_pr_auc=average_precision_score(y, pr),
                     st_tn=tn, st_fp=fp, st_fn=fn, st_tp=tp,
                     st_top1=obs_metrics(p, pr)["top1"], st_n_rows=len(p))
        for k in ("st_undirected_top1", "route_continuity", "route_breaks", "train_time_s",
                  "infer_ms_per_1k", "n_train_rows", "label_source"):
            r[k] = c.get(k, np.nan)
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("cv_top1_mean", ascending=False)
    out.to_csv(base / f"metrics_summary{sfx}.csv", index=False)
    print("wrote", base / f"metrics_summary{sfx}.csv")


if __name__ == "__main__":
    main()
