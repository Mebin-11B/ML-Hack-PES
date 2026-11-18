"""Order-preserving alignment of GPS fixes to positions in the reference route."""
from __future__ import annotations

import numpy as np
from .geo import LocalProjector


def align_route_to_track(trip, skip_penalty: float = 1.0):
    """Return per-fix route arc and distance using monotone Viterbi over route positions.

    Prefix minima keep the transition O(R) per observation; int32 pointers cap memory.
    """
    if trip.route is None:
        raise ValueError("trip must be loaded with with_route=True")
    route = np.asarray(trip.route, dtype=np.int64)
    n, r = len(trip.track), len(route)
    if not n or not r:
        return np.full(n, -1, np.int64), np.full(n, np.inf)
    proj = LocalProjector.from_trip(trip)
    nx, ny = proj.fwd(trip.nodes[:, 0], trip.nodes[:, 1])
    node_xy = np.column_stack([nx, ny])
    x, y = proj.fwd(trip.track[:, 0], trip.track[:, 1])
    f, t = trip.arcs[route].T
    ax, ay = node_xy[f].T; bx, by = node_xy[t].T
    dx, dy = bx-ax, by-ay
    den = np.maximum(dx*dx+dy*dy, 1.0)
    pos = np.arange(r, dtype=np.float64)
    back = np.empty((n, r), dtype=np.int32)
    prev = None
    for i in range(n):
        frac = np.clip(((x[i]-ax)*dx + (y[i]-ay)*dy)/den, 0, 1)
        d = np.hypot(x[i]-(ax+frac*dx), y[i]-(ay+frac*dy)).astype(np.float32)
        if prev is None:
            prev = d.astype(np.float64)
            back[i].fill(-1)
        else:
            z = prev - skip_penalty*pos
            # Prefix argmin from running minima; ties can use the later equivalent state.
            prefix = np.minimum.accumulate(z)
            arg = np.maximum.accumulate(np.where(z <= prefix + 1e-12, np.arange(r), 0)).astype(np.int32)
            prev = d + prefix + skip_penalty*pos
            back[i] = arg
    j = int(np.argmin(prev)); path = np.empty(n, dtype=np.int32)
    for i in range(n-1, -1, -1):
        path[i] = j
        if i: j = int(back[i, j])
    # Recompute selected distances in O(n), avoiding a second n×R allocation.
    q = path
    ddx, ddy = dx[q], dy[q]
    frac = np.clip(((x-ax[q])*ddx + (y-ay[q])*ddy) / np.maximum(ddx*ddx+ddy*ddy, 1.0), 0, 1)
    gd = np.hypot(x-(ax[q]+frac*ddx), y-(ay[q]+frac*ddy))
    return route[path].astype(np.int64), gd.astype(np.float64)
