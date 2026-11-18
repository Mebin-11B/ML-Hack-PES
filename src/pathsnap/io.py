"""Trip loading (M1). Frozen interface used by everyone.

Files per trip folder (whitespace separated, NO header, CRLF line endings):
  .track  lon lat t     (longitude FIRST)
  .nodes  lon lat       (row index = node_id)
  .arcs   from to       (row index = arc_id, directed)
  .route  arc_id        (one per line, ordered ground truth; NOT one per GPS point)

`pd.read_csv(sep=r"\\s+")` treats \\r\\n as whitespace, so CRLF is handled for free.
`with_route=False` is the default so prediction code can never see ground truth by accident.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray  # (n,3) lon, lat, t   float64
    nodes: np.ndarray  # (N,2) lon, lat      float64
    arcs: np.ndarray   # (A,2) from_node, to_node  int64
    route: np.ndarray | None = None  # (R,) int64 arc ids; None unless with_route=True


def _find(trip_dir: str, ext: str) -> str:
    """Locate '<trip_id>.<ext>' inside the trip folder (falls back to any *.ext)."""
    tid = os.path.basename(os.path.normpath(trip_dir))
    p = os.path.join(trip_dir, f"{tid}.{ext}")
    if os.path.exists(p):
        return p
    hits = sorted(glob.glob(os.path.join(trip_dir, f"*.{ext}")))
    if not hits:
        raise FileNotFoundError(f"no *.{ext} file in {trip_dir}")
    return hits[0]


def _read_table(path: str, ncols: int) -> np.ndarray:
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None)
    except pd.errors.EmptyDataError:
        return np.empty((0, ncols), dtype=np.float64)
    arr = df.to_numpy(dtype=np.float64)
    if arr.shape[1] < ncols:
        raise ValueError(f"{path}: expected >= {ncols} columns, got {arr.shape[1]}")
    return arr[:, :ncols]


def load_trip(trip_dir: str, with_route: bool = False) -> Trip:
    trip_dir = os.path.normpath(trip_dir)
    tid = os.path.basename(trip_dir)
    track = _read_table(_find(trip_dir, "track"), 3)
    nodes = _read_table(_find(trip_dir, "nodes"), 2)
    arcs = _read_table(_find(trip_dir, "arcs"), 2).astype(np.int64)
    route = None
    if with_route:
        route = _read_table(_find(trip_dir, "route"), 1).ravel().astype(np.int64)
    return Trip(trip_id=tid, track=track, nodes=nodes, arcs=arcs, route=route)


def list_trips(split_dir: str) -> list[str]:
    """Sorted trip folder paths inside a split folder (discovered at runtime, never hardcoded)."""
    out = []
    for name in sorted(os.listdir(split_dir)):
        d = os.path.join(split_dir, name)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.track")):
            out.append(d)
    return out
