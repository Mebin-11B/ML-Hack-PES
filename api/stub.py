"""M4 - Nearest-arc stub predictor + approximate scorer.

Purpose: lets the API and the evaluator UI be built and tested end-to-end (over ZeroTier)
before the real classifier + HMM pipeline (M3's `pathsnap.pipeline.PathSnapPredictor`)
exists. It returns exactly the same dict shape as `PathSnapPredictor.predict_trip`.

This module is self-contained on purpose (it does not depend on M1/M3 code), so it keeps
working while those modules are still being written. It is NOT the submitted model.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import shapely

EARTH_R = 6_371_008.8  # mean Earth radius in metres


def _read(path: str, dtype=float) -> np.ndarray:
    # whitespace separated, no header; regex separator also swallows CRLF line endings
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype).to_numpy()


def _trip_files(trip_dir: str) -> dict[str, str]:
    tid = os.path.basename(os.path.normpath(trip_dir))
    return {ext: os.path.join(trip_dir, f"{tid}.{ext}") for ext in ("track", "nodes", "arcs", "route")}


class _LocalXY:
    """Local equirectangular projection around the trip centroid (metres). Good to <0.1% at
    city scale; the real pipeline uses M1's UTM projector."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = np.radians(1.0) * EARTH_R * np.cos(np.radians(lat0))
        self.ky = np.radians(1.0) * EARTH_R

    def fwd(self, lon, lat):
        return np.c_[(np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky]


def _route_polyline(arcs: np.ndarray, nodes: np.ndarray, route_arcs: list[int]) -> list[list[float]]:
    """[[lat, lon], ...] along the arc sequence (consecutive duplicate vertices removed)."""
    pts: list[list[float]] = []
    for a in route_arcs:
        for n in arcs[a]:
            p = [float(nodes[n, 1]), float(nodes[n, 0])]
            if not pts or pts[-1] != p:
                pts.append(p)
    return pts


def _collapse(seq) -> list[int]:
    out: list[int] = []
    for a in seq:
        a = int(a)
        if a >= 0 and (not out or out[-1] != a):
            out.append(a)
    return out


class StubPredictor:
    name = "stub-nearest-arc"

    @classmethod
    def load(cls, artifacts_dir: str = "artifacts") -> "StubPredictor":
        return cls()

    def predict_trip(self, trip_dir: str) -> dict:
        t0 = time.perf_counter()
        f = _trip_files(trip_dir)
        track = _read(f["track"])                 # lon, lat, t
        nodes = _read(f["nodes"])                 # lon, lat
        arcs = _read(f["arcs"], dtype=np.int64)   # from, to   (.route deliberately NOT read)
        t_load = time.perf_counter()

        proj = _LocalXY(track[:, 0].mean(), track[:, 1].mean())
        nxy = proj.fwd(nodes[:, 0], nodes[:, 1])
        lines = shapely.linestrings(np.stack([nxy[arcs[:, 0]], nxy[arcs[:, 1]]], axis=1))
        tree = shapely.STRtree(lines)
        pts = shapely.points(proj.fwd(track[:, 0], track[:, 1]))
        t_index = time.perf_counter()

        # nearest arc per point (ties -> first hit)
        (pi, ai) = tree.query_nearest(pts, all_matches=False)
        obs_arc = np.full(len(track), -1, dtype=np.int64)
        obs_arc[pi] = ai
        t_match = time.perf_counter()

        route_arcs = _collapse(obs_arc)
        ms = lambda a, b: round((b - a) * 1000, 1)
        return {
            "trip_id": os.path.basename(os.path.normpath(trip_dir)),
            "obs_arc": obs_arc.tolist(),
            "obs_arc_clf": obs_arc.tolist(),
            "route_arcs": route_arcs,
            "route_latlon": _route_polyline(arcs, nodes, route_arcs),
            "gps_latlon": np.c_[track[:, 1], track[:, 0]].round(7).tolist(),
            "timings_ms": {
                "load": ms(t0, t_load),
                "candidates": ms(t_load, t_index),
                "features": 0.0,
                "classifier": ms(t_index, t_match),
                "viterbi": 0.0,
                "total": ms(t0, t_match),
            },
        }


def stub_score_trip(trip_dir: str, pred: dict) -> dict:
    """Approximate server-side scoring used only until M3's `pathsnap.scoring.score_trip` lands.

    Per-point reference label = nearest arc *among the reference route arcs* (approximation of
    M3's order-preserving alignment). Reads .route here only - never in the predictor.
    """
    f = _trip_files(trip_dir)
    keys = ["top1_hmm", "top1_clf", "top1_hmm_undirected", "route_precision", "route_recall",
            "route_f1", "continuity_hmm", "continuity_clf", "n_breaks_hmm", "n_valid_points"]
    if not os.path.exists(f["route"]):
        return {k: None for k in keys} | {"note": "no reference route available"}

    track = _read(f["track"])
    nodes = _read(f["nodes"])
    arcs = _read(f["arcs"], dtype=np.int64)
    route = _read(f["route"], dtype=np.int64).ravel()

    proj = _LocalXY(track[:, 0].mean(), track[:, 1].mean())
    nxy = proj.fwd(nodes[:, 0], nodes[:, 1])
    uniq = np.unique(route)
    lines = shapely.linestrings(np.stack([nxy[arcs[uniq, 0]], nxy[arcs[uniq, 1]]], axis=1))
    tree = shapely.STRtree(lines)
    pts = shapely.points(proj.fwd(track[:, 0], track[:, 1]))
    (pi, li), dist = tree.query_nearest(pts, all_matches=False, return_distance=True)
    gt = np.full(len(track), -1, dtype=np.int64)
    gd = np.full(len(track), np.inf)
    gt[pi], gd[pi] = uniq[li], dist
    valid = gd <= 30.0

    twin = {(int(a), int(b)): i for i, (a, b) in enumerate(arcs)}
    rev = np.array([twin.get((int(b), int(a)), -1) for a, b in arcs[np.maximum(gt, 0)]])

    def top1(p, undirected=False):
        p = np.asarray(p)
        ok = p == gt
        if undirected:
            ok |= (p == rev) & (rev >= 0)
        return float(ok[valid].mean()) if valid.any() else None

    def continuity(seq):
        s = _collapse(seq)
        if len(s) < 2:
            return 1.0, 0
        conn = arcs[np.array(s[:-1]), 1] == arcs[np.array(s[1:]), 0]
        return float(conn.mean()), int((~conn).sum())

    pr = set(pred["route_arcs"])
    rf = set(int(a) for a in route)
    tp = len(pr & rf)
    prec = tp / len(pr) if pr else 0.0
    rec = tp / len(rf) if rf else 0.0
    c_h, b_h = continuity(pred["obs_arc"])
    c_c, _ = continuity(pred["obs_arc_clf"])
    return {
        "top1_hmm": top1(pred["obs_arc"]),
        "top1_clf": top1(pred["obs_arc_clf"]),
        "top1_hmm_undirected": top1(pred["obs_arc"], undirected=True),
        "route_precision": round(prec, 4),
        "route_recall": round(rec, 4),
        "route_f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "continuity_hmm": round(c_h, 4),
        "continuity_clf": round(c_c, 4),
        "n_breaks_hmm": b_h,
        "n_valid_points": int(valid.sum()),
        "note": "approximate stub scoring (nearest reference-route arc labels)",
    }
