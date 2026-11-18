"""Per-trip metric projection (M1).

Why: 1 degree of longitude shrinks with cos(latitude) and trips are worldwide, so distances in
raw degrees are meaningless. We project each trip to the UTM zone of its centroid (metres).
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer


class LocalProjector:
    def __init__(self, lon0: float, lat0: float):
        zone = int(np.floor((lon0 + 180.0) / 6.0) % 60) + 1
        self.epsg = (32600 if lat0 >= 0 else 32700) + zone  # 326xx north / 327xx south
        self._f = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self._i = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        self.lon0, self.lat0 = float(lon0), float(lat0)

    @classmethod
    def from_trip(cls, trip) -> "LocalProjector":
        """Projector centred on the mean of the GPS track (never uses the route)."""
        return cls(float(np.mean(trip.track[:, 0])), float(np.mean(trip.track[:, 1])))

    def fwd(self, lon, lat):
        x, y = self._f.transform(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
        return np.asarray(x), np.asarray(y)

    def inv(self, x, y):
        lon, lat = self._i.transform(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return np.asarray(lon), np.asarray(lat)


def bearing_deg(dx, dy) -> np.ndarray:
    """Compass bearing of a displacement: 0 = north, clockwise, range [0, 360)."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0
