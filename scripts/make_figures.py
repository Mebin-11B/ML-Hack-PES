#!/usr/bin/env python
"""Report / review figures for M1 (all produced from data + code, saved to results/m1/figs/).

  python scripts/make_figures.py --split train                     # needs data/processed/train_candidates.parquet (or --parquet)
  python scripts/make_figures.py --split student_test
  python scripts/make_figures.py --split train --trips 00000033 00000054 --pred pred.json   # red predicted route on the static map

Figures:
  feature_dist_pos_neg.png     distributions of key features: true arc (label=1) vs other candidates (label=0)
  feature_univariate_auc.png   how well each single feature separates true arc from the rest (+ CSV)
  feature_correlation.png      feature correlation heatmap
  nearest_arc_disagree.png     per trip: how often the plain nearest arc != labelled arc (why we need ML + HMM)
  candidate_example_<trip>.png one observation: all candidates around it, true arc green (nearest arc is NOT the true one)
  static_map_<trip>.png        network (grey) + raw GPS (blue) + reference route (green) [+ predicted route (red)]; works offline
Only rows with label_valid == True are used for label-based plots.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from sklearn.metrics import roc_auc_score

from pathsnap.candidates import RoadIndex, track_xy
from pathsnap.geo import LocalProjector
from pathsnap.io import list_trips, load_trip

KEY_FEATURES = ["dist_m", "dist_minus_min", "heading_cos", "win2_mean_dist", "win5_mean_dist", "win5_rel",
                "win5_heading_cos", "speed_mps", "dist_z", "arc_len_m", "n_cands", "dist_to_end_m"]
POS, NEG = "#1a9a1a", "#e07b00"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  wrote", path)


def load_table(parquet, max_rows=400_000, seed=0):
    df = pd.read_parquet(parquet)
    full = df
    v = df[df["label_valid"]]
    if len(v) > max_rows:
        v = v.sample(max_rows, random_state=seed)
    return full, v


def fig_feature_dists(v, feats, out):
    feats = [f for f in KEY_FEATURES if f in v.columns]
    n = len(feats)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows))
    for ax, f in zip(np.ravel(axes), feats):
        lo, hi = np.percentile(v[f], [1, 99])
        bins = np.linspace(lo, hi, 40) if hi > lo else 20
        ax.hist(v.loc[v.label == 0, f].clip(lo, hi), bins=bins, alpha=.6, color=NEG, density=True, label="other candidates")
        ax.hist(v.loc[v.label == 1, f].clip(lo, hi), bins=bins, alpha=.6, color=POS, density=True, label="true arc")
        ax.set_title(f, fontsize=10); ax.set_yticks([])
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    np.ravel(axes)[0].legend(fontsize=8)
    fig.suptitle("Feature distributions: true arc vs other candidates (valid labels, 1-99 percentile)", y=1.0)
    _save(fig, out / "feature_dist_pos_neg.png")


def fig_univariate_auc(v, feats, out):
    rows = []
    for f in feats:
        if f in v.columns and v[f].nunique() > 1:
            a = roc_auc_score(v["label"], v[f])
            rows.append(dict(feature=f, auc=a, strength=max(a, 1 - a), direction="higher = more likely true arc" if a >= .5 else "lower = more likely true arc"))
    r = pd.DataFrame(rows).sort_values("strength", ascending=True)
    r.sort_values("strength", ascending=False).to_csv(out / "feature_univariate_auc.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(r) + 1.2))
    ax.barh(r["feature"], r["strength"], color="#3b6fd4")
    ax.axvline(.5, color="k", lw=.8)
    ax.set(xlim=(.45, 1.0), xlabel="single-feature ROC-AUC (direction-free) for 'is the true arc'")
    _save(fig, out / "feature_univariate_auc.png")


def fig_corr(v, feats, out):
    cols = [f for f in feats if f in v.columns and v[f].nunique() > 1] + ["label"]
    c = v[cols].corr()
    fig, ax = plt.subplots(figsize=(0.45 * len(cols) + 3, 0.45 * len(cols) + 2.5))
    im = ax.imshow(c, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=.8)
    ax.set_title("Feature correlation (valid rows)")
    _save(fig, out / "feature_correlation.png")


def fig_disagree(summary_csv, out):
    if not Path(summary_csv).exists():
        print("  (skip nearest_arc_disagree: no", summary_csv, ")")
        return
    s = pd.read_csv(summary_csv, dtype={"trip_id": str}).sort_values("nearest_arc_disagree")
    fig, ax = plt.subplots(figsize=(max(6, .25 * len(s)), 3.8))
    x = np.arange(len(s))
    ax.bar(x, s["nearest_arc_disagree"] * 100, label="nearest arc != label (directed)", color="#d9534f")
    ax.bar(x, s["nearest_arc_disagree_undirected"] * 100, label="... even ignoring direction (twin counted ok)", color="#f0ad4e")
    ax.set_xticks([]); ax.set(xlabel="trips", ylabel="% of valid observations")
    src = ",".join(sorted(s["label_source"].unique()))
    ax.set_title(f"Why 'nearest arc' is not enough (labels: {src})"); ax.legend(fontsize=8)
    _save(fig, out / "nearest_arc_disagree.png")


def _arc_xy(ri):
    return np.stack([np.column_stack([ri.ax, ri.ay]), np.column_stack([ri.bx, ri.by])], axis=1)  # (A,2,2)


def fig_candidate_example(df, trip_dir, out, window_m=90.0):
    tid = Path(trip_dir).name
    d = df[(df.trip_id == tid) & df.label_valid]
    ex = d[(d.label == 1) & (d.dist_rank > 0) & (d.n_cands >= 8) & (d.dist_m <= 25) & (d.arc_len_m >= 15)]
    if ex.empty:
        ex = d[(d.label == 1) & (d.dist_rank > 0) & (d.n_cands >= 6)]
    if ex.empty:
        print(f"  (no example obs where nearest != true arc in {tid})")
        return
    ex = ex.sort_values(["n_cands", "obs_idx"], ascending=[False, True]).head(max(5, len(ex) // 10))
    row = ex.iloc[len(ex) // 2]
    oi = int(row.obs_idx)
    trip = load_trip(trip_dir, with_route=False)
    proj = LocalProjector.from_trip(trip)
    ri = RoadIndex(trip, proj)
    xy = track_xy(trip, proj)
    p = xy[oi]
    segs = _arc_xy(ri) - p  # local metres centred on the observation
    xy = xy - p
    p = np.zeros(2)
    ax_, ay_, bx_, by_ = (segs[:, 0, 0], segs[:, 0, 1], segs[:, 1, 0], segs[:, 1, 1])
    dx_, dy_ = bx_ - ax_, by_ - ay_
    L2 = dx_ ** 2 + dy_ ** 2
    tt = np.clip(np.where(L2 > 0, -(ax_ * dx_ + ay_ * dy_) / np.where(L2 > 0, L2, 1), 0), 0, 1)
    dist_to_p = np.hypot(ax_ + tt * dx_, ay_ + tt * dy_)
    near = np.flatnonzero(dist_to_p < window_m)
    cand = d[d.obs_idx == oi].sort_values("dist_rank")
    cset = dict(zip(cand.arc_id, zip(cand.label, cand.dist_rank, cand.dist_m)))
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for a in near:
        s, e = segs[a]
        v = e - s
        L = np.hypot(*v) or 1.0
        off = np.array([v[1], -v[0]]) / L * 1.8  # draw each direction slightly to its right so twins are visible
        s2, e2 = s + off, e + off
        if a in cset:
            lab, rk, dm = cset[a]
            col, lw, z = (POS, 3.2, 5) if lab == 1 else (NEG, 2.0, 4)
        else:
            col, lw, z = "#bbbbbb", 1.0, 1
        ax.annotate("", xy=e2, xytext=s2, arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=9), zorder=z)
        if a in cset:
            c = (s2 + e2) / 2
            ax.text(c[0], c[1], f"{cset[a][1]:.0f}", fontsize=8, color=col, zorder=6, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=.15", fc="white", ec="none", alpha=.8))
    lo, hi = max(oi - 5, 0), min(oi + 6, len(xy))
    ax.plot(xy[lo:hi, 0], xy[lo:hi, 1], "o-", color="#1f5fff", ms=4, lw=.8, zorder=7, label="GPS fixes i-5..i+5")
    ax.plot(*p, "*", color="red", ms=16, zorder=8, label=f"observation {oi}")
    ax.plot([], [], color=POS, lw=3, label="true arc"); ax.plot([], [], color=NEG, lw=2, label="other candidates (number = distance rank)")
    ax.plot([], [], color="#bbbbbb", lw=1, label="other arcs (not candidates)")
    ax.set_aspect("equal"); ax.set(xlim=(-window_m, window_m), ylim=(-window_m, window_m),
                                   xlabel="east of observation (m)", ylabel="north of observation (m)")
    ax.set_title(f"trip {tid}: nearest arc (rank 0) is NOT the true arc here; twins drawn side by side", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, out / f"candidate_example_{tid}.png")


def fig_static_map(trip_dir, out, pred_latlon=None, max_points=6000):
    trip = load_trip(trip_dir, with_route=True)
    proj = LocalProjector.from_trip(trip)
    ri = RoadIndex(trip, proj)
    xy = track_xy(trip, proj)
    segs = _arc_xy(ri)
    r = segs[trip.route].reshape(-1, 2)
    pad = 250.0
    x0, x1 = min(xy[:, 0].min(), r[:, 0].min()) - pad, max(xy[:, 0].max(), r[:, 0].max()) + pad
    y0, y1 = min(xy[:, 1].min(), r[:, 1].min()) - pad, max(xy[:, 1].max(), r[:, 1].max()) + pad
    mid = segs.mean(axis=1)
    inb = (mid[:, 0] > x0) & (mid[:, 0] < x1) & (mid[:, 1] > y0) & (mid[:, 1] < y1) & ((ri.twin < 0) | (np.arange(ri.n_arcs) < ri.twin))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_collection(LineCollection(segs[inb], colors="#b5b5b5", linewidths=.5, label="road network"))
    st = max(1, len(xy) // max_points)
    ax.scatter(xy[::st, 0], xy[::st, 1], s=3, c="#1f5fff", label="raw GPS", zorder=3)
    ax.add_collection(LineCollection(segs[trip.route], colors=POS, linewidths=2.2, label="reference route", zorder=2))
    if pred_latlon:
        pl = np.array(pred_latlon)
        px, py = proj.fwd(pl[:, 1], pl[:, 0])
        ax.plot(px, py, color="#e02020", lw=1.6, label="predicted route", zorder=4)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set(xlabel="east (m, local UTM)", ylabel="north (m, local UTM)", title=f"trip {trip.trip_id}: {len(xy)} GPS points, {ri.n_arcs} arcs")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.legend(loc="best", fontsize=8, markerscale=3)
    _save(fig, out / f"static_map_{trip.trip_id}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--data-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "m1" / "figs"))
    ap.add_argument("--trips", nargs="*", default=None)
    ap.add_argument("--pred", default=None, help="JSON with route_latlon (predictor output) for the static map")
    ap.add_argument("--skip-features", action="store_true", help="only maps/examples")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pq = Path(a.parquet) if a.parquet else ROOT / "data" / "processed" / f"{a.split}_candidates.parquet"
    if not pq.exists():
        alt = ROOT / "data" / "processed" / f"{a.split}_sample_candidates.parquet"
        pq = alt if alt.exists() else pq
    print(f"[figs] table: {pq}")
    full, v = load_table(pq)
    feats = [d["name"] for d in json.load(open(ROOT / "artifacts" / "feature_list.json"))]
    feats = [f for f in feats if f in v.columns]
    if not a.skip_features:
        fig_feature_dists(v, feats, out)
        fig_univariate_auc(v, feats, out)
        fig_corr(v, feats, out)
        fig_disagree(ROOT / "results" / "m1" / f"trip_summary_{a.split}.csv", out)

    trips = {Path(d).name: d for d in list_trips(str(Path(a.data_root) / a.split))}
    chosen = a.trips
    if not chosen:
        summ = ROOT / "results" / "m1" / f"trip_summary_{a.split}.csv"
        if summ.exists():
            s = pd.read_csv(summ, dtype={"trip_id": str})
            fl = s[s["flagged"]]["trip_id"].tolist()[:1]
            ok = s[~s["flagged"] & (s["n_points"] < 6000)]
            ok = ok.iloc[(ok["n_points"] - ok["n_points"].median()).abs().argsort()]["trip_id"].tolist()[: 3 - len(fl)]
            chosen = fl + ok
        else:
            chosen = list(trips)[:3]
    pred = json.load(open(a.pred))["route_latlon"] if a.pred else None
    for tid in chosen:
        if tid not in trips:
            print("  (trip not found:", tid, ")"); continue
        fig_static_map(trips[tid], out, pred)
        fig_candidate_example(full, trips[tid], out)
    print("[figs] done ->", out)


if __name__ == "__main__":
    main()
