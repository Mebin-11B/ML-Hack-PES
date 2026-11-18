#!/usr/bin/env python
"""EDA for PathSnap (M1). Every number/plot comes from code and is saved to results/m1/.

  python scripts/run_eda.py --splits train student_test            # everything
  python scripts/run_eda.py --splits train --skip-candidates        # quick (no miss-rate curve)

The notebook notebooks/01_eda.ipynb calls the same functions so outputs appear inline.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.candidates import RoadIndex, get_labels, nearest_route_arc, track_xy
from pathsnap.geo import LocalProjector
from pathsnap.io import list_trips, load_trip

RADII = [20, 30, 50, 75, 100]
FILES = ["track", "nodes", "arcs", "route"]


# --------------------------------------------------------------------------------------------
# per-trip statistics (inventory, sampling, validity, extent, network, route)
# --------------------------------------------------------------------------------------------
def _trip_stats(split: str, trip_dir: str, flag_median_m: float):
    trip = load_trip(trip_dir, with_route=True)
    tid = trip.trip_id
    tr, nodes, arcs, route = trip.track, trip.nodes, trip.arcs, trip.route
    n, N, A = len(tr), len(nodes), len(arcs)
    sizes = {f: sum(p.stat().st_size for p in Path(trip_dir).glob(f"*.{f}")) for f in FILES}
    row = dict(split=split, trip_id=tid, n_files=len(list(Path(trip_dir).iterdir())),
               bytes_track=sizes["track"], bytes_nodes=sizes["nodes"], bytes_arcs=sizes["arcs"], bytes_route=sizes["route"])

    # sampling
    t = tr[:, 2]
    dt = np.diff(t)
    row.update(n_points=n, duration_s=float(t[-1] - t[0]) if n else 0.0,
               share_1s_steps=float(np.mean(np.isclose(dt, 1.0))) if len(dt) else np.nan,
               n_gaps_gt10s=int((dt > 10).sum()), n_gaps_gt60s=int((dt > 60).sum()),
               max_gap_s=float(dt.max()) if len(dt) else 0.0)

    # missing / invalid
    row.update(n_nan=int(np.isnan(tr).sum() + np.isnan(nodes).sum()), n_dup_timestamps=int((dt == 0).sum()),
               n_non_increasing_t=int((dt < 0).sum()),
               n_coords_out_of_range=int(((np.abs(tr[:, 0]) > 180) | (np.abs(tr[:, 1]) > 90)).sum()),
               n_bad_node_refs=int(((arcs < 0) | (arcs >= N)).any(axis=1).sum()),
               n_bad_route_refs=int(((route < 0) | (route >= A)).sum()))

    proj = LocalProjector.from_trip(trip)
    x, y = proj.fwd(tr[:, 0], tr[:, 1])
    step = np.hypot(np.diff(x), np.diff(y))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dt > 0, step / dt, 0.0)
    row.update(n_speed_gt60=int((v > 60).sum()), max_step_speed_mps=float(v.max()) if len(v) else 0.0,
               bbox_w_km=float((x.max() - x.min()) / 1000), bbox_h_km=float((y.max() - y.min()) / 1000),
               centroid_lon=float(tr[:, 0].mean()), centroid_lat=float(tr[:, 1].mean()))

    if row["n_bad_node_refs"] or row["n_bad_route_refs"]:  # cannot build a graph on broken references
        return row, dt.astype(np.float32), np.array([])

    # road network
    ri = RoadIndex(trip, proj)
    f, tt = arcs[:, 0], arcs[:, 1]
    g = csr_matrix((np.ones(A), (f, tt)), shape=(N, N))
    nw, lw = connected_components(g, directed=True, connection="weak")
    ns, ls = connected_components(g, directed=True, connection="strong")
    used = np.zeros(N, bool)
    used[f] = True
    used[tt] = True
    row.update(n_nodes=N, n_arcs=A, weak_components=int(nw), strong_components=int(ns),
               largest_weak_share=float(np.bincount(lw).max() / N), largest_strong_share=float(np.bincount(ls).max() / N),
               isolated_nodes=int((~used).sum()), twin_ratio=float(ri.has_twin.mean()),
               zero_len_arcs=int((ri.arc_len == 0).sum()), arc_len_mean_m=float(ri.arc_len.mean()),
               arc_len_median_m=float(np.median(ri.arc_len)), arc_len_p95_m=float(np.percentile(ri.arc_len, 95)))

    # route
    ra = arcs[route]
    breaks = int((ra[:-1, 1] != ra[1:, 0]).sum()) if len(route) > 1 else 0
    _, rd = nearest_route_arc(ri, np.column_stack([x, y]), route)
    row.update(route_arcs=len(route), route_unique_arcs=int(len(np.unique(route))),
               route_repeated_arcs=int(len(route) - len(np.unique(route))), route_km=float(ri.arc_len[route].sum() / 1000),
               route_breaks=breaks, gps_route_dist_median_m=float(np.median(rd)), gps_route_dist_p95_m=float(np.percentile(rd, 95)),
               gps_route_dist_max_m=float(rd.max()), pct_gps_gt50m_from_route=float((rd > 50).mean() * 100),
               flagged_offset=bool(np.median(rd) > flag_median_m))
    return row, dt.astype(np.float32), rd.astype(np.float32)


def collect_trip_stats(splits, data_root, out_dir, flag_median_m=50.0, max_trips=None, n_jobs=1):
    rows, dts, rds = [], {}, {}
    jobs = []
    for sp in splits:
        trips = list_trips(str(Path(data_root) / sp))
        if max_trips:
            trips = trips[:max_trips]
        jobs += [(sp, d) for d in trips]
    if n_jobs == 1:
        results = [_trip_stats(sp, d, flag_median_m) for sp, d in jobs]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(delayed(_trip_stats)(sp, d, flag_median_m) for sp, d in jobs)
    for (sp, d), (row, dt, rd) in zip(jobs, results):
        rows.append(row)
        dts.setdefault(sp, []).append(dt)
        rds[(sp, row["trip_id"])] = rd
    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "trip_level_stats.csv", index=False)
    dts = {sp: np.concatenate(v) for sp, v in dts.items()}
    return df, dts, rds


def summarise(df: pd.DataFrame, out_dir) -> dict:
    """Inventory summary + quality summary per split (tables for the report)."""
    out = Path(out_dir)
    inv = df.groupby("split").agg(
        trips=("trip_id", "count"), files_per_trip=("n_files", "median"),
        total_points=("n_points", "sum"), points_min=("n_points", "min"), points_median=("n_points", "median"),
        points_max=("n_points", "max"), nodes_min=("n_nodes", "min"), nodes_max=("n_nodes", "max"),
        arcs_min=("n_arcs", "min"), arcs_max=("n_arcs", "max"),
        mb_total=("bytes_track", lambda s: 0.0)).drop(columns="mb_total")
    inv["size_mb_total"] = (df.groupby("split")[["bytes_track", "bytes_nodes", "bytes_arcs", "bytes_route"]].sum().sum(axis=1) / 1e6)
    qual = df.groupby("split").agg(
        nan_values=("n_nan", "sum"), duplicate_timestamps=("n_dup_timestamps", "sum"),
        non_increasing_t=("n_non_increasing_t", "sum"), coords_out_of_range=("n_coords_out_of_range", "sum"),
        steps_over_60mps=("n_speed_gt60", "sum"), trips_with_gap_gt10s=("n_gaps_gt10s", lambda s: int((s > 0).sum())),
        trips_with_gap_gt60s=("n_gaps_gt60s", lambda s: int((s > 0).sum())), max_gap_s=("max_gap_s", "max"),
        mean_share_1s_steps=("share_1s_steps", "mean"), trips_with_route_breaks=("route_breaks", lambda s: int((s > 0).sum())),
        trips_with_repeated_arcs=("route_repeated_arcs", lambda s: int((s > 0).sum())),
        trips_flagged_offset=("flagged_offset", "sum"), mean_twin_ratio=("twin_ratio", "mean"),
        trips_with_isolated_nodes=("isolated_nodes", lambda s: int((s > 0).sum())),
        trips_not_weakly_connected=("weak_components", lambda s: int((s > 1).sum())))
    inv.to_csv(out / "inventory_summary.csv")
    qual.to_csv(out / "quality_summary.csv")
    df.loc[df["flagged_offset"] == True, ["split", "trip_id", "gps_route_dist_median_m", "gps_route_dist_p95_m"]] \
        .to_csv(out / "offset_trips_by_rule.csv", index=False)
    return dict(inventory=inv, quality=qual)


# --------------------------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------------------------
def _save(fig, out_dir, name):
    p = Path(out_dir) / name
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_all(df, dts, rds, out_dir, flag_median_m=50.0):
    out = Path(out_dir)
    paths = []
    # 1) points per trip + network size
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for sp, g in df.groupby("split"):
        ax[0].hist(g["n_points"], bins=20, alpha=.6, label=sp)
        ax[1].hist(np.log10(g["n_nodes"].dropna()), bins=20, alpha=.6, label=sp)
        ax[2].hist(np.log10(g["n_arcs"].dropna()), bins=20, alpha=.6, label=sp)
    ax[0].set(xlabel="GPS points per trip", ylabel="trips"); ax[1].set(xlabel="log10(nodes per trip)"); ax[2].set(xlabel="log10(arcs per trip)")
    ax[0].legend()
    paths.append(_save(fig, out, "network_size.png"))

    # 2) dt histogram (log y)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for sp, dt in dts.items():
        ax.hist(np.clip(dt, 0, 20), bins=np.arange(0, 20.5, 0.5), alpha=.6, label=f"{sp} (last bin = >=20 s)")
    ax.set_yscale("log"); ax.set(xlabel="time step dt (s), clipped at 20 s", ylabel="count (log)"); ax.legend()
    paths.append(_save(fig, out, "dt_hist.png"))

    # 3) GPS -> route distance per trip (box plot) : exposes offset trips
    for sp in df["split"].unique():
        keys = [k for k in rds if k[0] == sp and len(rds[k])]
        if not keys:
            continue
        order = sorted(keys, key=lambda k: np.median(rds[k]))
        fig, ax = plt.subplots(figsize=(max(7, 0.18 * len(order)), 4))
        ax.boxplot([rds[k] for k in order], showfliers=False, whis=(5, 95))
        ax.axhline(flag_median_m, color="r", ls="--", lw=1, label=f"flag rule: median > {flag_median_m:.0f} m")
        ax.set_xticks([]); ax.set(xlabel=f"{sp} trips (sorted by median)", ylabel="GPS -> nearest route arc (m)"); ax.legend()
        paths.append(_save(fig, out, f"gps_to_route_boxplot_{sp}.png"))

    # 4) world scatter of trip centroids
    fig, ax = plt.subplots(figsize=(8, 4))
    for sp, g in df.groupby("split"):
        ax.scatter(g["centroid_lon"], g["centroid_lat"], s=18, alpha=.8, label=sp)
    ax.set(xlabel="longitude", ylabel="latitude", xlim=(-180, 180), ylim=(-90, 90)); ax.grid(alpha=.3); ax.legend()
    paths.append(_save(fig, out, "world_centroids.png"))

    # 5) gaps + route length
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    g = df[df["split"] == df["split"].iloc[0]]
    ax[0].hist(np.log10(g["max_gap_s"].clip(lower=1)), bins=20); ax[0].set(xlabel="log10(max gap per trip, s)", ylabel="trips")
    ax[1].hist(g["route_km"].dropna(), bins=20); ax[1].set(xlabel="route length (km)", ylabel="trips")
    paths.append(_save(fig, out, "gaps_and_route_length.png"))
    return paths


# --------------------------------------------------------------------------------------------
# candidate miss rate vs radius (needs labels -> train)
# --------------------------------------------------------------------------------------------
def _trip_candidate_counts(trip_dir, radii, k_min, k_max, valid_gt_m, flag_median_m):
    trip = load_trip(trip_dir, with_route=True)
    proj = LocalProjector.from_trip(trip)
    ri = RoadIndex(trip, proj)
    xy = track_xy(trip, proj)
    gt, gd, src = get_labels(trip, proj, ri, xy)
    _, rd = nearest_route_arc(ri, xy, trip.route)
    flagged = np.median(rd) > flag_median_m
    valid = (gd <= valid_gt_m) & (not flagged)
    n = len(gt)
    # one query at the largest radius with no cap; smaller radii are derived by filtering
    c = ri.query(xy, radius=max(radii), k_min=k_min, k_max=10**9)
    oi, ai, d = c["obs_idx"].to_numpy(), c["arc_id"].to_numpy(), c["dist_m"].to_numpy()
    starts = np.flatnonzero(np.r_[True, oi[1:] != oi[:-1]])
    sizes = np.diff(np.r_[starts, len(oi)])
    rank = np.arange(len(oi)) - np.repeat(starts, sizes)
    is_gt = ai == gt[oi]
    res = {}
    for R in radii:
        keep_rule = ((d <= R + 1e-6) | (rank < k_min)) & (rank < k_max)
        keep_rad = d <= R + 1e-6
        gt_rule = np.zeros(n, bool); gt_rule[oi[keep_rule & is_gt]] = True
        gt_rad = np.zeros(n, bool); gt_rad[oi[keep_rad & is_gt]] = True
        cnt_rule = np.bincount(oi[keep_rule], minlength=n)
        res[R] = dict(valid=int(valid.sum()), miss_radius_only=int((~gt_rad[valid]).sum()),
                      miss_rule=int((~gt_rule[valid]).sum()), cands_rule=int(cnt_rule.sum()), n_obs=n,
                      hist=np.bincount(cnt_rule, minlength=k_max + 1)[: k_max + 1] if R == 50 else None)
    return res, src


def candidate_stats(data_root, out_dir, split="train", radii=RADII, k_min=8, k_max=30, valid_gt_m=30.0,
                    flag_median_m=50.0, max_trips=None, n_jobs=1):
    trips = list_trips(str(Path(data_root) / split))
    if max_trips:
        trips = trips[:max_trips]
    if n_jobs == 1:
        outs = [_trip_candidate_counts(d, radii, k_min, k_max, valid_gt_m, flag_median_m) for d in trips]
    else:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_jobs)(delayed(_trip_candidate_counts)(d, radii, k_min, k_max, valid_gt_m, flag_median_m) for d in trips)
    rows, hist = [], np.zeros(k_max + 1, dtype=np.int64)
    for R in radii:
        v = sum(o[0][R]["valid"] for o in outs); n = sum(o[0][R]["n_obs"] for o in outs)
        rows.append(dict(split=split, radius_m=R, valid_obs=v,
                         miss_rate_radius_only=sum(o[0][R]["miss_radius_only"] for o in outs) / max(v, 1),
                         miss_rate_radius_plus_knn=sum(o[0][R]["miss_rule"] for o in outs) / max(v, 1),
                         mean_cands_per_obs=sum(o[0][R]["cands_rule"] for o in outs) / max(n, 1),
                         k_min=k_min, k_max=k_max, label_source=",".join(sorted({o[1] for o in outs}))))
    for o in outs:
        hist += o[0][50]["hist"] if 50 in radii and o[0][50]["hist"] is not None else 0
    df = pd.DataFrame(rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "candidate_stats.csv", index=False)
    pd.DataFrame({"n_cands": np.arange(k_max + 1), "n_obs": hist}).to_csv(out / "cands_per_obs_hist.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(df.radius_m, df.miss_rate_radius_only * 100, "o-", label="radius only")
    ax[0].plot(df.radius_m, df.miss_rate_radius_plus_knn * 100, "s-", label=f"radius + {k_min} nearest (cap {k_max})")
    ax[0].set(xlabel="radius R (m)", ylabel="candidate miss rate (% of valid obs)", title=f"{split}: gt arc missing from candidates"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax2 = ax[0].twinx(); ax2.plot(df.radius_m, df.mean_cands_per_obs, "g--", label="mean cands/obs"); ax2.set_ylabel("mean candidates / obs", color="g")
    ax[1].bar(np.arange(k_max + 1), hist); ax[1].set(xlabel="candidates per observation (R=50 m rule)", ylabel="observations")
    _save(fig, out, "miss_rate_vs_radius.png")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--splits", nargs="+", default=["train", "student_test"])
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "m1"))
    ap.add_argument("--flag-median-m", type=float, default=50.0)
    ap.add_argument("--max-trips", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--skip-candidates", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    df, dts, rds = collect_trip_stats(a.splits, a.data_root, a.out_dir, a.flag_median_m, a.max_trips, a.n_jobs)
    s = summarise(df, a.out_dir)
    plot_all(df, dts, rds, a.out_dir, a.flag_median_m)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(s["inventory"].T, "\n"); print(s["quality"].T, "\n")
    print("offset trips by rule:\n", df.loc[df["flagged_offset"] == True, ["split", "trip_id", "gps_route_dist_median_m"]].to_string(index=False))
    if not a.skip_candidates and "train" in a.splits:
        print("\ncandidate stats (train):\n", candidate_stats(a.data_root, a.out_dir, "train", max_trips=a.max_trips,
                                                              flag_median_m=a.flag_median_m, n_jobs=a.n_jobs).to_string(index=False))
    print(f"\nwrote CSV + PNG into {a.out_dir}")


if __name__ == "__main__":
    main()
