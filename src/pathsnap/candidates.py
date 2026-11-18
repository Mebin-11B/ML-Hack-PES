"""Candidate generation (M1).

For every GPS observation find the arcs it could be on:
  * all arcs within `radius` metres, PLUS
  * always the `k_min` nearest arcs (so far-off points still get candidates), capped at `k_max`.

Arcs are straight segments node->node, so point-to-arc distance is computed with plain numpy
(vectorised). A shapely STRtree does the spatial search (needed: up to ~300k arcs per trip).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shapely

from .geo import LocalProjector, bearing_deg


class RoadIndex:
    """Built once per trip; reused by features, API and the HMM (M3)."""

    def __init__(self, trip, projector: LocalProjector):
        self.projector = projector
        nx, ny = projector.fwd(trip.nodes[:, 0], trip.nodes[:, 1])
        self.node_xy = np.column_stack([nx, ny])
        arcs = trip.arcs
        self.arcs = arcs
        self.n_nodes = len(trip.nodes)
        self.n_arcs = len(arcs)
        f, t = arcs[:, 0], arcs[:, 1]
        self.ax, self.ay = self.node_xy[f, 0], self.node_xy[f, 1]
        self.bx, self.by = self.node_xy[t, 0], self.node_xy[t, 1]
        dx, dy = self.bx - self.ax, self.by - self.ay
        self.arc_len = np.hypot(dx, dy)
        self.arc_bearing = bearing_deg(dx, dy)  # degrees, 0 = north
        self.arc_bearing_rad = np.radians(self.arc_bearing)

        coords = np.stack([np.column_stack([self.ax, self.ay]), np.column_stack([self.bx, self.by])], axis=1)
        self.lines = shapely.linestrings(coords)  # (A,) LineStrings in metres
        self.tree = shapely.STRtree(self.lines)

        # node degree = number of distinct undirected neighbours (intersection complexity)
        N = max(self.n_nodes, 1)
        a, b = np.minimum(f, t), np.maximum(f, t)
        m = a != b
        key = np.unique(a[m] * N + b[m])
        u, v = key // N, key % N
        self.node_deg = np.bincount(u, minlength=N) + np.bincount(v, minlength=N)
        self.from_deg = self.node_deg[f]
        self.to_deg = self.node_deg[t]

        # twin = arc running the opposite way between the same nodes (two-way road => duplicate geometry)
        fwd_key = f * N + t
        rev_key = t * N + f
        order = np.argsort(fwd_key, kind="stable")
        sk = fwd_key[order]
        pos = np.clip(np.searchsorted(sk, rev_key), 0, max(len(sk) - 1, 0))
        found = (sk[pos] == rev_key) & (f != t) if len(sk) else np.zeros(0, bool)
        self.twin = np.where(found, order[pos], -1).astype(np.int64)
        self.has_twin = (self.twin >= 0)

    # ---- geometry helpers -------------------------------------------------------------------
    def dist_to_arcs(self, px, py, arc_ids):
        """Distance (m) from points to the given arcs (pairwise), plus projection fraction and foot point."""
        ax, ay, bx, by = self.ax[arc_ids], self.ay[arc_ids], self.bx[arc_ids], self.by[arc_ids]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        safe = np.where(L2 > 0, L2, 1.0)
        t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / safe, 0.0)
        t = np.clip(t, 0.0, 1.0)
        qx, qy = ax + t * dx, ay + t * dy
        return np.hypot(px - qx, py - qy), t, qx, qy

    def _pairs(self, xy, radius):
        ii, aa = self.tree.query(shapely.points(xy), predicate="dwithin", distance=radius)
        return ii, aa

    # ---- main query -------------------------------------------------------------------------
    def query(self, xy: np.ndarray, radius: float = 50.0, k_min: int = 8, k_max: int = 30,
              max_search: float = 5000.0) -> pd.DataFrame:
        xy = np.asarray(xy, dtype=np.float64)
        n = len(xy)
        if n == 0 or self.n_arcs == 0:
            return pd.DataFrame({c: [] for c in ["obs_idx", "arc_id", "dist_m", "proj_frac", "proj_x", "proj_y"]})
        oi, ai = self._pairs(xy, radius)

        # observations with fewer than k_min arcs inside the radius: grow the search radius (x2) until
        # k_min arcs are found (this is the "always k nearest" rule, done with vectorised range queries)
        if k_min > 0:
            cnt = np.bincount(oi, minlength=n)
            need = np.flatnonzero(cnt < k_min)
            if need.size:
                mask = np.zeros(n, dtype=bool)
                mask[need] = True
                keep_base = ~mask[oi]
                extra_o, extra_a = [], []
                r = radius
                while need.size and r < max_search:
                    r = min(r * 2.0, max_search)
                    o2, a2 = self._pairs(xy[need], r)
                    c2 = np.bincount(o2, minlength=need.size)
                    ok = c2 >= k_min
                    if r >= max_search:
                        ok[:] = True  # give up growing; take whatever was found
                    sel = ok[o2]
                    extra_o.append(need[o2[sel]])
                    extra_a.append(a2[sel])
                    need = need[~ok]
                oi = np.concatenate([oi[keep_base]] + extra_o)
                ai = np.concatenate([ai[keep_base]] + extra_a)

        # guarantee >= 1 candidate per observation (nearest arc, even if > max_search away)
        cnt = np.bincount(oi, minlength=n)
        empty = np.flatnonzero(cnt == 0)
        if empty.size:
            idx = self.tree.query_nearest(shapely.points(xy[empty]), all_matches=False)
            oi = np.concatenate([oi, empty[idx[0]]])
            ai = np.concatenate([ai, idx[1]])

        d, frac, qx, qy = self.dist_to_arcs(xy[oi, 0], xy[oi, 1], ai)
        order = np.lexsort((ai, d, oi))  # by obs, then distance, then arc_id (deterministic ties)
        oi, ai, d, frac, qx, qy = oi[order], ai[order], d[order], frac[order], qx[order], qy[order]

        starts = np.flatnonzero(np.r_[True, oi[1:] != oi[:-1]])
        sizes = np.diff(np.r_[starts, len(oi)])
        rank = np.arange(len(oi)) - np.repeat(starts, sizes)
        keep = ((d <= radius + 1e-6) | (rank < k_min)) & (rank < k_max)
        return pd.DataFrame({
            "obs_idx": oi[keep].astype(np.int64), "arc_id": ai[keep].astype(np.int64),
            "dist_m": d[keep], "proj_frac": frac[keep], "proj_x": qx[keep], "proj_y": qy[keep],
        }).reset_index(drop=True)


def track_xy(trip, projector: LocalProjector) -> np.ndarray:
    x, y = projector.fwd(trip.track[:, 0], trip.track[:, 1])
    return np.column_stack([x, y])


def generate_candidates(trip, projector: LocalProjector | None = None, road_index: RoadIndex | None = None,
                        radius: float = 50.0, k_min: int = 8, k_max: int = 30) -> pd.DataFrame:
    """One row per (observation, candidate arc). Columns: trip_id, obs_idx, t, arc_id, dist_m,
    proj_frac, proj_x, proj_y (sorted by obs_idx, dist_m)."""
    projector = projector or LocalProjector.from_trip(trip)
    road_index = road_index or RoadIndex(trip, projector)
    df = road_index.query(track_xy(trip, projector), radius=radius, k_min=k_min, k_max=k_max)
    df.insert(0, "trip_id", trip.trip_id)
    df.insert(2, "t", trip.track[df["obs_idx"].to_numpy(), 2])
    return df


# ---- helpers used for labels / data-quality (they read the route, so only for training-side code) ----
def nearest_route_arc(road_index: RoadIndex, xy: np.ndarray, route_arcs: np.ndarray):
    """For each GPS point: nearest arc AMONG the route's arcs and the distance to it (m).
    Order-agnostic, so it is only a fallback label / a data-quality measure (flags offset trips)."""
    ra = np.unique(route_arcs)
    tree = shapely.STRtree(road_index.lines[ra])
    idx, dist = tree.query_nearest(shapely.points(xy), return_distance=True, all_matches=False)
    arc = np.empty(len(xy), dtype=np.int64)
    dd = np.empty(len(xy), dtype=np.float64)
    arc[idx[0]] = ra[idx[1]]
    dd[idx[0]] = dist
    return arc, dd


def get_labels(trip, projector: LocalProjector, road_index: RoadIndex, xy: np.ndarray):
    """Per-observation ground-truth arc. Uses M3's order-preserving alignment when it exists,
    otherwise the documented fallback (nearest arc among route arcs). Returns (gt_arc, gt_dist_m, source)."""
    if trip.route is None:
        raise ValueError("get_labels needs a trip loaded with with_route=True")
    try:
        from pathsnap.labels import align_route_to_track  # M3
    except ImportError:
        align_route_to_track = None
    if align_route_to_track is not None:
        try:
            gt, gd = align_route_to_track(trip)
            return np.asarray(gt, dtype=np.int64), np.asarray(gd, dtype=np.float64), "M3_alignment"
        except Exception as e:  # never let a label bug kill the whole build; say so loudly
            warnings.warn(f"[{trip.trip_id}] M3 align_route_to_track failed ({e!r}); using fallback labels")
    gt, gd = nearest_route_arc(road_index, xy, trip.route)
    return gt, gd, "fallback_nearest_route_arc"
