"""M2 — plots from saved CSVs/JSONs only (no numbers invented here).

  python scripts/plot_models.py --tag v0            # reads results/m2/model_comparison_v0.csv
  python scripts/plot_models.py --toy               # reads results/m2/toy/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def bar(ax, names, vals, err=None, ylabel="", title=""):
    ax.bar(names, vals, yerr=err, capsize=4, color="#4C72B0")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=8)


def load_preds(pred_dir: Path, stem: str) -> pd.DataFrame | None:
    for ext in (".parquet", ".csv.gz"):
        p = pred_dir / f"{stem}{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--chosen", default=None, help="model for confusion matrix (default: best CV top1)")
    args = ap.parse_args()
    base = ROOT / "results/m2" / ("toy" if args.toy else "")
    sfx = f"_{args.tag}" if args.tag else ""
    out = base / "plots"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(base / f"model_comparison{sfx}.csv").sort_values("cv_top1_mean", ascending=False)
    names = df["model"].tolist()

    # 1. CV macro-F1 and CV Top-1 with fold std
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    bar(axs[0], names, df.cv_macro_f1_mean, df.cv_macro_f1_std, "macro F1", "CV macro-F1 (GroupKFold by trip)")
    bar(axs[1], names, df.cv_top1_mean, df.cv_top1_std, "Top-1", "CV observation Top-1 accuracy")
    fig.tight_layout(); fig.savefig(out / f"cv_scores{sfx}.png", dpi=150); plt.close(fig)

    # 2. student_test Top-1 & macro-F1 grouped
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, df.st_top1, w, label="Top-1")
    ax.bar(x + w / 2, df.st_macro_f1, w, label="macro F1")
    ax.set_xticks(x, names, rotation=30); ax.set_title("student_test"); ax.legend()
    lo = min(df.st_top1.min(), df.st_macro_f1.min())
    ax.set_ylim(max(0, lo - 0.05), 1)
    fig.tight_layout(); fig.savefig(out / f"student_test_scores{sfx}.png", dpi=150); plt.close(fig)

    # 3. log loss + inference time
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    bar(axs[0], names, df.cv_logloss, None, "log loss (lower better)", "CV log loss")
    bar(axs[1], names, df.infer_ms_per_1k, None, "ms per 1k rows", "Inference time (student_test)")
    axs[1].set_yscale("log")
    fig.tight_layout(); fig.savefig(out / f"logloss_infer{sfx}.png", dpi=150); plt.close(fig)

    # 4. confusion matrix of chosen model (from saved config)
    chosen = args.chosen or names[0]
    cfg = base / "configs" / f"{chosen}{sfx}.json"
    if cfg.exists():
        cm = np.array(json.loads(cfg.read_text())["st_confusion"])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"{chosen} — student_test (thr 0.5)")
        fig.tight_layout(); fig.savefig(out / f"confusion_{chosen}{sfx}.png", dpi=150); plt.close(fig)

    # 5. real training histories only
    for h in sorted((base / "histories").glob(f"*{sfx}.json")):
        name = h.stem[: -len(sfx)] if sfx else h.stem
        if sfx == "" and "_" in name:
            continue
        hist = json.loads(h.read_text())
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist["train"], label="train")
        if "valid" in hist:
            ax.plot(hist["valid"], label="validation (inner trip split)")
        if "valid_accuracy" in hist and hist["valid_accuracy"]:
            ax2 = ax.twinx(); ax2.plot(hist["valid_accuracy"], color="C2", label="valid accuracy")
            ax2.set_ylabel("valid accuracy"); ax2.legend(loc="center right")
        ax.set_xlabel("iteration / epoch"); ax.set_ylabel(hist["metric"]); ax.set_title(f"{name} training history")
        ax.legend(); fig.tight_layout(); fig.savefig(out / f"history_{name}{sfx}.png", dpi=150); plt.close(fig)

    # 6. reliability curves for top-3 models (valid student_test rows)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name in names[:3]:
        p = load_preds(base / "preds", f"st_{name}{sfx}")
        if p is None:
            continue
        p = p[p.label_valid.astype(bool)]
        fy, mp = calibration_curve(p.label, p.p, n_bins=15, strategy="quantile")
        ax.plot(mp, fy, marker="o", ms=3, label=name)
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("fraction positive")
    ax.set_title("Reliability (student_test)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / f"reliability{sfx}.png", dpi=150); plt.close(fig)
    print("plots ->", out)


if __name__ == "__main__":
    main()
