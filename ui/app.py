"""M4 - PathSnap evaluator UI (runs on Laptop 2).

Talks to the API on Laptop 1 over HTTP only (normally via its ZeroTier IP). It never imports
the prediction code and never sees reference routes - only predictions + aggregate metrics.

Run:
    streamlit run ui/app.py --server.port 8501
    (optional) PATHSNAP_API=http://<laptop1-zerotier-ip>:8000 streamlit run ui/app.py
"""
from __future__ import annotations

import os
import time

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap Evaluator", page_icon="🛰️", layout="wide")

DEFAULT_API = os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000")
GPS_COLOR, ROUTE_COLOR = "#1f6feb", "#d62728"

if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None


# ---------------- sidebar: connection ----------------
with st.sidebar:
    st.header("API connection")
    api = st.text_input("API base URL (Laptop 1)", value=DEFAULT_API,
                        help="e.g. http://10.147.17.10:8000 — Laptop 1's ZeroTier IP").rstrip("/")
    try:
        h = requests.get(f"{api}/health", timeout=5).json()
        st.success(f"Connected · model: {h.get('model')} · trips: {h.get('n_trips')}")
        connected = True
    except Exception as e:
        st.error(f"Cannot reach {api}/health\n\n{type(e).__name__}")
        connected = False
    if st.button("Re-check connection"):
        st.rerun()
    st.caption("Predictions and accuracy are computed on Laptop 1. "
               "This UI only sends the trip ID and displays the response.")

st.title("PathSnap — GPS Map-Matching Evaluator")

if not connected:
    st.info("Set the API URL in the sidebar (Laptop 1's ZeroTier IP, port 8000).")
    st.stop()


# ---------------- dataset + trip selection ----------------
DATASET_LABELS = {"evaluator_test": "Evaluator test trips (final evaluation)",
                  "student_test": "Student test trips (our development split)"}


@st.cache_data(ttl=30, show_spinner=False)
def fetch_datasets(base: str) -> list[dict]:
    try:
        r = requests.get(f"{base}/datasets", timeout=10)
        r.raise_for_status()
        return [d for d in r.json()["datasets"] if d["n_trips"] > 0]
    except Exception:  # older API without /datasets -> single default trip set
        return [{"name": "evaluator_test", "n_trips": None}]


@st.cache_data(ttl=30, show_spinner=False)
def fetch_trips(base: str, dataset: str) -> pd.DataFrame:
    r = requests.get(f"{base}/trips", params={"dataset": dataset}, timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json()["trips"])


available = fetch_datasets(api)
if not available:
    st.warning("The API reports no trips. Put the evaluation trips in the API's trip folders.")
    st.stop()
names = [d["name"] for d in available]
dataset = st.radio("Trip set", names, horizontal=True,
                   format_func=lambda n: DATASET_LABELS.get(n, n) + next(
                       (f" — {d['n_trips']} trips" for d in available if d["name"] == n and d["n_trips"]), ""))

try:
    trips = fetch_trips(api, dataset)
except Exception as e:
    st.error(f"Could not load trips: {e}")
    st.stop()

if trips.empty:
    st.warning(f"No trips in '{dataset}'.")
    st.stop()

left, right = st.columns([1, 2])
with left:
    st.subheader(f"{DATASET_LABELS.get(dataset, dataset).split(' (')[0]} ({len(trips)})")
    st.dataframe(trips.rename(columns={"trip_id": "Trip ID", "n_points": "GPS points",
                                       "duration_s": "Duration (s)"}),
                 hide_index=True, width="stretch")
    trip_id = st.selectbox("Select a trip", trips["trip_id"].tolist(), key=f"trip_{dataset}")
    run = st.button("▶ Run map-matching", type="primary", width="stretch")

if run:
    with st.spinner(f"Laptop 1 is matching trip {trip_id} ({dataset}) …"):
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{api}/predict", json={"trip_id": trip_id, "dataset": dataset}, timeout=180)
            r.raise_for_status()
            res = r.json()
            res["_roundtrip_s"] = time.perf_counter() - t0
            res.setdefault("dataset", dataset)
            st.session_state.result = res
            m = res.get("metrics", {})
            st.session_state.history.insert(0, {
                "time": time.strftime("%H:%M:%S"), "trip set": dataset, "trip": trip_id,
                "points": res["n_points"],
                "top1 (HMM)": m.get("top1_hmm"), "top1 (classifier)": m.get("top1_clf"),
                "route F1": m.get("route_f1"), "continuity": m.get("continuity_hmm"),
                "server s": round(res["timings_ms"].get("server_total", 0) / 1000, 2)})
        except Exception as e:
            st.session_state.result = None
            st.error(f"Prediction request failed: {e}")


# ---------------- results ----------------
def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


res = st.session_state.result
with right:
    if res is None:
        st.info("Pick a trip and press **Run map-matching**.")
    else:
        m = res.get("metrics", {})
        st.subheader(f"Trip {res['trip_id']} · {res.get('dataset', '')} · model: {res.get('model')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Top-1 accuracy (HMM)", pct(m.get("top1_hmm")))
        c2.metric("Top-1 (classifier only)", pct(m.get("top1_clf")))
        c3.metric("Route F1", pct(m.get("route_f1")))
        c4.metric("Route continuity", pct(m.get("continuity_hmm")))
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Route precision", pct(m.get("route_precision")))
        c6.metric("Route recall", pct(m.get("route_recall")))
        c7.metric("GPS points", f"{res['n_points']:,}")
        c8.metric("Server time", f"{res['timings_ms'].get('server_total', 0) / 1000:.2f} s")

        gps, route = res["gps_latlon"], res["route_latlon"]
        lats = [p[0] for p in gps] + [p[0] for p in route]
        lons = [p[1] for p in gps] + [p[1] for p in route]
        fmap = folium.Map(tiles="OpenStreetMap", control_scale=True)
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

        g_layer = folium.FeatureGroup(name="Raw GPS observations", show=True)
        step = max(1, len(gps) // 4000)  # keep the browser responsive on very long trips
        for lat, lon in gps[::step]:
            folium.CircleMarker([lat, lon], radius=2, color=GPS_COLOR, fill=True,
                                fill_opacity=0.7, weight=0).add_to(g_layer)
        g_layer.add_to(fmap)

        r_layer = folium.FeatureGroup(name="Predicted route", show=True)
        if len(route) >= 2:
            folium.PolyLine(route, color=ROUTE_COLOR, weight=5, opacity=0.85,
                            tooltip="Predicted route").add_to(r_layer)
        r_layer.add_to(fmap)

        folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
        folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)
        legend = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
                  f'padding:8px 12px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);font-size:13px">'
                  f'<span style="color:{GPS_COLOR}">●</span> Raw GPS &nbsp; '
                  f'<span style="color:{ROUTE_COLOR}">━━</span> Predicted route</div>')
        fmap.get_root().html.add_child(folium.Element(legend))
        st_folium(fmap, height=560, use_container_width=True, returned_objects=[])

        with st.expander("Server timing breakdown & raw metrics"):
            st.json({"timings_ms": res.get("timings_ms"), "metrics": m,
                     "round_trip_s": round(res.get("_roundtrip_s", 0), 2)})

if st.session_state.history:
    st.subheader("Runs this session")
    st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, width="stretch")
