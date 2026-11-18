#!/usr/bin/env python
"""Folium maps (M1): road network (grey) + raw GPS (blue) + reference route (green) [+ predicted route (red)].

  python scripts/make_maps.py --split train                       # auto-picks 3 trips (1 offset-flagged if any)
  python scripts/make_maps.py --split train --trips 00000033 00000054 --png
  python scripts/make_maps.py --split student_test --trips 00000005 --pred pred.json   # pred.json = predictor output (route_latlon)

Output: results/maps/eda_<trip>.html (or pred_<trip>.html). --png tries a Playwright screenshot (needs internet
for map tiles); otherwise just open the HTML in a browser and take a screenshot for the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import folium
import numpy as np
import pandas as pd

from pathsnap.candidates import RoadIndex
from pathsnap.geo import LocalProjector
from pathsnap.io import list_trips, load_trip

LEGEND = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white; padding: 8px 12px;
            border: 1px solid #888; border-radius: 6px; font-size: 13px; line-height: 1.5;">
<b>{title}</b><br>
<span style="color:#999">&#9632;</span> road network (arcs)<br>
<span style="color:#1f5fff">&#9679;</span> raw GPS points<br>
<span style="color:#1a9a1a">&#9632;</span> reference route (ground truth)<br>
{pred}
</div>"""


def build_map(trip, out_path, pred_latlon=None, max_arcs=25000, max_points=2500):
    proj = LocalProjector.from_trip(trip)
    ri = RoadIndex(trip, proj)
    lon, lat = trip.track[:, 0], trip.track[:, 1]
    nodes, arcs = trip.nodes, trip.arcs
    # clip the network to the trip's bounding box (+ margin) so the HTML stays small
    pad = 0.003
    box = (min(lon.min(), nodes[arcs[trip.route][:, 0], 0].min()) - pad, max(lon.max(), nodes[arcs[trip.route][:, 1], 0].max()) + pad,
           min(lat.min(), nodes[arcs[trip.route][:, 0], 1].min()) - pad, max(lat.max(), nodes[arcs[trip.route][:, 1], 1].max()) + pad)
    f, t = arcs[:, 0], arcs[:, 1]
    inb = ((nodes[f, 0] >= box[0]) & (nodes[f, 0] <= box[1]) & (nodes[f, 1] >= box[2]) & (nodes[f, 1] <= box[3]) &
           (nodes[t, 0] >= box[0]) & (nodes[t, 0] <= box[1]) & (nodes[t, 1] >= box[2]) & (nodes[t, 1] <= box[3]))
    draw = inb & ((ri.twin < 0) | (np.arange(len(arcs)) < ri.twin))  # draw each two-way road once
    ids = np.flatnonzero(draw)
    if len(ids) > max_arcs:
        print(f"  note: {len(ids)} arcs in view, drawing a deterministic subset of {max_arcs}")
        ids = ids[np.linspace(0, len(ids) - 1, max_arcs).astype(int)]
    seg = lambda a: [[nodes[arcs[a, 0], 1], nodes[arcs[a, 0], 0]], [nodes[arcs[a, 1], 1], nodes[arcs[a, 1], 0]]]
    net = [seg(a) for a in ids]

    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=14, control_scale=True)
    folium.PolyLine(net, color="#9a9a9a", weight=1, opacity=.8).add_to(folium.FeatureGroup("road network").add_to(m))
    g_gps = folium.FeatureGroup("raw GPS").add_to(m)
    stride = max(1, len(lon) // max_points)
    for la, lo in zip(lat[::stride], lon[::stride]):
        folium.CircleMarker([float(la), float(lo)], radius=2, color="#1f5fff", fill=True, fill_opacity=.8, weight=0).add_to(g_gps)
    folium.PolyLine([seg(a) for a in trip.route], color="#1a9a1a", weight=4, opacity=.85).add_to(folium.FeatureGroup("reference route").add_to(m))
    pred_html = ""
    if pred_latlon:
        folium.PolyLine(pred_latlon, color="#e02020", weight=3, opacity=.9).add_to(folium.FeatureGroup("predicted route").add_to(m))
        pred_html = '<span style="color:#e02020">&#9632;</span> predicted route (HMM)<br>'
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(title=f"trip {trip.trip_id}", pred=pred_html)))
    m.save(str(out_path))
    return out_path


def screenshot(html_path):
    try:
        from playwright.sync_api import sync_playwright
        png = Path(html_path).with_suffix(".png")
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=2)
            pg.goto(Path(html_path).resolve().as_uri())
            pg.wait_for_timeout(4000)
            pg.screenshot(path=str(png))
            b.close()
        print("  png:", png)
    except Exception as e:
        print(f"  (no PNG: {e!r}) -> open the HTML in a browser and screenshot it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--trips", nargs="*", default=None, help="trip ids (folder names)")
    ap.add_argument("--pred", default=None, help="JSON with key route_latlon [[lat,lon],...] (PathSnapPredictor output)")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "maps"))
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()

    split_dir = Path(a.data_root) / a.split
    all_trips = {Path(d).name: d for d in list_trips(str(split_dir))}
    if a.trips:
        chosen = a.trips
    else:
        summ = ROOT / "results" / "m1" / f"trip_summary_{a.split}.csv"
        if summ.exists():
            s = pd.read_csv(summ, dtype={"trip_id": str})
            flagged = s[s["flagged"]]["trip_id"].tolist()[:1]
            ok = s[~s["flagged"] & (s["n_points"] < 6000)]
            ok = ok.iloc[(ok["n_points"] - ok["n_points"].median()).abs().argsort()]["trip_id"].tolist()[: 3 - len(flagged)]
            chosen = flagged + ok
        else:
            chosen = list(all_trips)[:3]
    pred = None
    if a.pred:
        pred = json.load(open(a.pred))["route_latlon"]
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for tid in chosen:
        trip = load_trip(all_trips[tid], with_route=True)
        out = Path(a.out_dir) / (f"pred_{tid}.html" if pred else f"eda_{tid}.html")
        print(f"[map] {tid}: {len(trip.track)} points, {len(trip.arcs)} arcs -> {out}")
        build_map(trip, out, pred_latlon=pred)
        if a.png:
            screenshot(out)


if __name__ == "__main__":
    main()
