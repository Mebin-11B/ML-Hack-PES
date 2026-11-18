#!/usr/bin/env python
"""Build the (observation, candidate) training table.  Owner: M1.

  python scripts/build_dataset.py --split train
  python scripts/build_dataset.py --split student_test
  python scripts/build_dataset.py --split train --max-trips 20 --max-points 6000     # v0 hand-off sample
  python scripts/build_dataset.py --split train --features v0                        # fastest, core features only

Output: data/processed/<split>_candidates.parquet  (or <split>_sample_candidates.parquet with --max-trips)
        results/m1/trip_summary_<split>.csv, results/m1/data_quality_flags.csv, artifacts/feature_list.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pathsnap.candidates import RoadIndex, generate_candidates, get_labels, nearest_route_arc, track_xy
from pathsnap.features import FEATURES, FEATURES_V0, add_features, feature_list
from pathsnap.geo import LocalProjector
from pathsnap.io import list_trips, load_trip

BASE_COLS = ["trip_id", "obs_idx", "t", "arc_id", "dist_m", "label", "label_valid", "gt_arc"]


def final_columns(version):
    feats = FEATURES if version == "v1" else FEATURES_V0
    return BASE_COLS + [f for f in feats if f not in BASE_COLS]


def finalize(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df[cols].copy()
    df["trip_id"] = df["trip_id"].astype(str)
    for c in ("obs_idx", "arc_id", "gt_arc"):
        df[c] = df[c].astype(np.int32)
    df["t"] = df["t"].astype(np.float64)
    df["label"] = df["label"].astype(np.int8)
    df["label_valid"] = df["label_valid"].astype(bool)
    for c in cols:
        if c not in BASE_COLS or c == "dist_m":
            df[c] = df[c].astype(np.float32)
    return df


def process_trip(trip_dir: str, split: str, cfg: dict):
    trip = load_trip(trip_dir, with_route=True)
    proj = LocalProjector.from_trip(trip)
    ri = RoadIndex(trip, proj)
    xy = track_xy(trip, proj)

    # --- labels (M3 alignment if pushed, else documented fallback) + rule-based quality flag
    gt, gt_d, source = get_labels(trip, proj, ri, xy)
    _, route_d = nearest_route_arc(ri, xy, trip.route)  # label-independent GPS->reference-route distance
    med_rd, p95_rd = float(np.median(route_d)), float(np.percentile(route_d, 95))
    flagged = med_rd > cfg["flag_median_m"]
    reason = f"median GPS->route distance {med_rd:.0f} m > {cfg['flag_median_m']:.0f} m (systematic offset)" if flagged else ""

    # --- candidates + features
    cands = generate_candidates(trip, proj, ri, radius=cfg["radius"], k_min=cfg["k_min"], k_max=cfg["k_max"])
    df = add_features(cands, trip, proj, ri, version=cfg["features"])
    oi = df["obs_idx"].to_numpy()
    df["gt_arc"] = gt[oi]
    df["label"] = (df["arc_id"].to_numpy() == gt[oi]).astype(np.int8)
    obs_valid = (gt_d <= cfg["valid_gt_m"]) & (not flagged)
    df["label_valid"] = obs_valid[oi]

    # --- per-trip summary (candidate miss rate, nearest-arc disagreement, ...)
    n = len(gt)
    has_gt = np.zeros(n, dtype=bool)
    has_gt[oi[df["label"].to_numpy() == 1]] = True
    nearest = df[df["dist_rank"] == 0].set_index("obs_idx")["arc_id"].reindex(np.arange(n)).to_numpy()
    twin_gt = ri.twin[gt]
    nv = max(int(obs_valid.sum()), 1)
    summ = dict(
        split=split, trip_id=trip.trip_id, n_points=n, n_arcs=len(trip.arcs), n_cands_rows=len(df),
        mean_cands_per_obs=len(df) / max(n, 1), label_source=source,
        median_route_dist_m=med_rd, p95_route_dist_m=p95_rd, median_gt_dist_m=float(np.median(gt_d)),
        flagged=bool(flagged), flag_reason=reason, n_valid_obs=int(obs_valid.sum()),
        cand_miss_rate=float(1.0 - has_gt[obs_valid].mean()) if obs_valid.any() else np.nan,
        nearest_arc_disagree=float((nearest[obs_valid] != gt[obs_valid]).sum() / nv),
        nearest_arc_disagree_undirected=float(((nearest[obs_valid] != gt[obs_valid]) & (nearest[obs_valid] != twin_gt[obs_valid])).sum() / nv),
    )
    return finalize(df, final_columns(cfg["features"])), summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="train | student_test (folder under --data-root)")
    ap.add_argument("--data-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--out", default=None, help="output file name (default <split>_candidates.parquet)")
    ap.add_argument("--max-trips", type=int, default=None, help="use N evenly spaced trips (hand-off sample)")
    ap.add_argument("--max-points", type=int, default=None, help="skip trips with more GPS points than this")
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--k-min", type=int, default=8)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--features", choices=["v0", "v1"], default="v1")
    ap.add_argument("--valid-gt-m", type=float, default=30.0, help="label_valid needs aligned-gt distance <= this")
    ap.add_argument("--flag-median-m", type=float, default=50.0, help="flag trip if median GPS->route dist > this")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--results-dir", default=str(ROOT / "results" / "m1"))
    ap.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    a = ap.parse_args()

    cfg = dict(radius=a.radius, k_min=a.k_min, k_max=a.k_max, features=a.features,
               valid_gt_m=a.valid_gt_m, flag_median_m=a.flag_median_m)
    trips = list_trips(str(Path(a.data_root) / a.split))
    if not trips:
        sys.exit(f"no trips found under {Path(a.data_root) / a.split}")
    if a.max_points:
        keep = []
        for d in trips:
            with open(next(Path(d).glob("*.track"))) as fh:
                if sum(1 for _ in fh) <= a.max_points:
                    keep.append(d)
        trips = keep
    if a.max_trips and len(trips) > a.max_trips:
        trips = [trips[i] for i in np.linspace(0, len(trips) - 1, a.max_trips).round().astype(int)]
    out_name = a.out or (f"{a.split}_sample_candidates.parquet" if a.max_trips else f"{a.split}_candidates.parquet")
    out_path = Path(a.out_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(a.results_dir).mkdir(parents=True, exist_ok=True)
    Path(a.artifacts_dir).mkdir(parents=True, exist_ok=True)
    print(f"[build] split={a.split} trips={len(trips)} features={a.features} -> {out_path}")

    t0 = time.time()
    if a.n_jobs == 1:
        gen = (process_trip(d, a.split, cfg) for d in trips)
    else:
        from joblib import Parallel, delayed
        gen = Parallel(n_jobs=a.n_jobs, return_as="generator")(delayed(process_trip)(d, a.split, cfg) for d in trips)

    writer, summaries, nrows = None, [], 0
    for k, (df, summ) in enumerate(gen, 1):
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table.cast(writer.schema))
        summaries.append(summ)
        nrows += len(df)
        print(f"  [{k}/{len(trips)}] {summ['trip_id']}: {summ['n_points']} pts, {len(df)} rows, "
              f"miss={summ['cand_miss_rate']:.4f}, flagged={summ['flagged']}, labels={summ['label_source']}  ({time.time()-t0:.0f}s)")
    if writer is not None:
        writer.close()

    res = pd.DataFrame(summaries)
    res.to_csv(Path(a.results_dir) / f"trip_summary_{a.split}.csv", index=False)
    flags_path = Path(a.results_dir) / "data_quality_flags.csv"
    new = res.loc[res["flagged"], ["split", "trip_id", "median_route_dist_m", "p95_route_dist_m", "flag_reason"]]
    if flags_path.exists():
        old = pd.read_csv(flags_path, dtype={"trip_id": str})
        new = pd.concat([old[old["split"] != a.split], new], ignore_index=True)
    new.to_csv(flags_path, index=False)
    with open(Path(a.artifacts_dir) / "feature_list.json", "w") as fh:
        json.dump(feature_list(a.features), fh, indent=2)

    print(f"[build] done: {nrows:,} rows, {len(res)} trips, flagged={int(res['flagged'].sum())}, "
          f"overall miss rate (valid obs)={np.average(res['cand_miss_rate'].fillna(0), weights=res['n_valid_obs'].clip(lower=1)):.4f}, "
          f"{time.time()-t0:.0f}s. size={out_path.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
