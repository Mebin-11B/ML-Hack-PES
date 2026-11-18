"""Feature engineering for (observation, candidate arc) pairs (M1).

All geometry is in metres in the trip's local UTM projection. No absolute coordinates are used as
features (trips are worldwide and would not generalise). Nothing here reads ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAP_S = 10.0              # dt above this = a gap; heading/windows never cross a gap
HEADING_MIN_DISP_M = 2.0  # centred-difference displacement below this => heading is noise => treat as unknown (0)

# (name, version, why)
FEATURE_DOC = [
    ("dist_m", "v0", "Perpendicular distance GPS point -> arc segment. Right arc is usually the closest."),
    ("dist_rank", "v0", "Rank of this candidate by distance within the observation (0 = nearest)."),
    ("dist_minus_min", "v0", "dist minus the nearest candidate's dist: relative closeness, comparable across noisy/clean trips."),
    ("dist_ratio", "v0", "dist / (min dist + 1): scale-free closeness."),
    ("n_cands", "v0", "Number of candidates for this observation (junction / dense-network indicator)."),
    ("heading_cos", "v0", "cos(GPS heading - arc bearing): +1 same direction, -1 reverse twin. Separates the two directed arcs of a two-way road."),
    ("heading_sin_abs", "v0", "|sin(heading diff)|: how perpendicular the motion is to the arc (cross streets at junctions)."),
    ("speed_mps", "v0", "Speed from the +-2 point window; lets the model learn heading is unreliable when slow."),
    ("arc_len_m", "v0", "Arc length: short arcs near junctions behave differently from long road segments."),
    ("proj_frac", "v0", "Where along the arc the point projects (0 start .. 1 end)."),
    ("dist_to_end_m", "v0", "Metres from the projection to the arc's downstream end (about to turn / leave the arc)."),
    ("has_twin", "v0", "Reverse arc exists (about 92% of arcs): direction matters for these."),
    ("win2_mean_dist", "v1", "Mean distance of points i-2..i+2 to this arc: temporal consistency, damps single-point noise."),
    ("win5_mean_dist", "v1", "Same over i-5..i+5: the true arc stays close for the whole window, wrong arcs drift away."),
    ("win5_rel", "v1", "win5_mean_dist minus the best candidate's win5_mean_dist for this observation (relative window fit)."),
    ("win5_heading_cos", "v1", "Mean heading agreement with this arc over the window: more robust than a single noisy heading."),
    ("from_deg", "v1", "Degree of the arc's start node (intersection complexity)."),
    ("to_deg", "v1", "Degree of the arc's end node (intersection complexity)."),
    ("dt_prev", "v1", "Seconds since previous fix (sampling irregularity)."),
    ("dt_next", "v1", "Seconds to next fix."),
    ("gap_flag", "v1", "1 if a neighbouring step is a gap (> 10 s): window/heading features are weaker there."),
    ("accel", "v1", "Acceleration (m/s^2) from the speed series: stops/starts happen at junctions."),
    ("dist_z", "v1", "dist / per-trip GPS-noise estimate (median nearest-arc distance, label-free): noise-normalised distance."),
]
FEATURES_V0 = [n for n, v, _ in FEATURE_DOC if v == "v0"]
FEATURES = [n for n, _, _ in FEATURE_DOC]  # v1 = everything


def feature_list(version: str = "v1"):
    names = FEATURES_V0 if version == "v0" else FEATURES
    why = {n: w for n, _, w in FEATURE_DOC}
    return [{"name": n, "why": why[n]} for n in names]


def point_kinematics(xy: np.ndarray, t: np.ndarray):
    """Per-GPS-point speed, heading (radians, 0=north, NaN-free), heading validity, accel, dt_prev/next, segment id.
    Heading/speed use a centred difference (i-2 -> i+2) clipped to the same gap-free segment."""
    n = len(t)
    dt = np.diff(t)
    seg = np.concatenate([[0], np.cumsum(dt > GAP_S)]).astype(np.int64)
    idx = np.arange(n)
    starts = np.flatnonzero(np.r_[True, np.diff(seg) != 0])
    ends = np.r_[starts[1:] - 1, n - 1]
    lens = ends - starts + 1
    seg_start, seg_end = np.repeat(starts, lens), np.repeat(ends, lens)
    i0, i1 = np.maximum(idx - 2, seg_start), np.minimum(idx + 2, seg_end)
    d = xy[i1] - xy[i0]
    dtt = t[i1] - t[i0]
    disp = np.hypot(d[:, 0], d[:, 1])
    speed = np.where(dtt > 0, disp / np.where(dtt > 0, dtt, 1.0), 0.0)
    heading = np.arctan2(d[:, 0], d[:, 1])
    hvalid = disp >= HEADING_MIN_DISP_M
    heading = np.where(hvalid, heading, 0.0)
    if n >= 2:
        accel = np.clip(np.gradient(speed, t), -10.0, 10.0)
        dt_prev = np.r_[dt[0], dt]
        dt_next = np.r_[dt, dt[-1]]
    else:
        accel, dt_prev, dt_next = np.zeros(n), np.zeros(n), np.zeros(n)
    return dict(speed=speed, heading=heading, hvalid=hvalid, accel=accel, dt_prev=dt_prev, dt_next=dt_next, seg=seg)


def add_features(cands: pd.DataFrame, trip, projector, road_index, version: str = "v1") -> pd.DataFrame:
    """Add model features to the candidate table (v0 = fast core block, v1 = + window/gap features)."""
    ri = road_index
    cands = cands.sort_values(["obs_idx", "dist_m", "arc_id"], kind="stable").reset_index(drop=True)
    x, y = projector.fwd(trip.track[:, 0], trip.track[:, 1])
    xy = np.column_stack([x, y])
    t = trip.track[:, 2]
    n = len(t)
    kin = point_kinematics(xy, t)

    oi = cands["obs_idx"].to_numpy()
    ai = cands["arc_id"].to_numpy()
    dist = cands["dist_m"].to_numpy()
    m = len(cands)
    out = cands.copy()

    # ---- distance-relative features (rows are sorted by obs then dist, so first row of a group = nearest)
    starts = np.flatnonzero(np.r_[True, oi[1:] != oi[:-1]])
    sizes = np.diff(np.r_[starts, m])
    rank = np.arange(m) - np.repeat(starts, sizes)
    mind = np.repeat(dist[starts], sizes)
    out["dist_rank"] = rank
    out["dist_minus_min"] = dist - mind
    out["dist_ratio"] = dist / (mind + 1.0)
    out["n_cands"] = np.repeat(sizes, sizes)

    # ---- heading / kinematics features
    bearing = ri.arc_bearing_rad[ai]
    hv = kin["hvalid"][oi]
    hdiff = kin["heading"][oi] - bearing
    out["heading_cos"] = np.where(hv, np.cos(hdiff), 0.0)
    out["heading_sin_abs"] = np.where(hv, np.abs(np.sin(hdiff)), 0.0)
    out["speed_mps"] = kin["speed"][oi]
    out["arc_len_m"] = ri.arc_len[ai]
    out["dist_to_end_m"] = (1.0 - cands["proj_frac"].to_numpy()) * ri.arc_len[ai]
    out["has_twin"] = ri.has_twin[ai].astype(np.float32)

    # ---- v1 block
    out["from_deg"] = ri.from_deg[ai]
    out["to_deg"] = ri.to_deg[ai]
    out["dt_prev"] = kin["dt_prev"][oi]
    out["dt_next"] = kin["dt_next"][oi]
    out["gap_flag"] = ((kin["dt_prev"][oi] > GAP_S) | (kin["dt_next"][oi] > GAP_S)).astype(np.float32)
    out["accel"] = kin["accel"][oi]
    nearest_per_obs = dist[starts]  # nearest-arc distance for every observation (starts are in obs order)
    sigma = max(float(np.median(nearest_per_obs)), 2.0) if len(nearest_per_obs) else 2.0
    out["dist_z"] = dist / sigma

    if version == "v1":
        # window consistency: distance of neighbours i-5..i+5 (same gap-free segment only) to THIS arc
        seg = kin["seg"]
        s2 = np.zeros(m); c2 = np.zeros(m)
        s5 = np.zeros(m); c5 = np.zeros(m)
        h5 = np.zeros(m); hc5 = np.zeros(m)
        for d_off in range(-5, 6):
            j = oi + d_off
            inb = (j >= 0) & (j < n)
            jc = np.clip(j, 0, n - 1)
            same = inb & (seg[jc] == seg[oi])
            dd = ri.dist_to_arcs(xy[jc, 0], xy[jc, 1], ai)[0]
            s5 += np.where(same, dd, 0.0)
            c5 += same
            hok = same & kin["hvalid"][jc]
            h5 += np.where(hok, np.cos(kin["heading"][jc] - bearing), 0.0)
            hc5 += hok
            if abs(d_off) <= 2:
                s2 += np.where(same, dd, 0.0)
                c2 += same
        out["win2_mean_dist"] = s2 / np.maximum(c2, 1)
        w5 = s5 / np.maximum(c5, 1)
        out["win5_mean_dist"] = w5
        out["win5_rel"] = w5 - np.repeat(np.minimum.reduceat(w5, starts), sizes)
        out["win5_heading_cos"] = h5 / np.maximum(hc5, 1)

    cols = FEATURES if version == "v1" else FEATURES_V0
    for c in cols:
        if c not in out.columns:
            raise KeyError(c)
    return out
