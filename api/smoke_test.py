"""M4 - End-to-end API rehearsal: call /health, /trips and /predict for every trip.

Run from Laptop 2 (over ZeroTier) or Laptop 1 (localhost):
    python api/smoke_test.py --api http://<laptop1-zerotier-ip>:8000
Writes results/api/latency_<host>.csv and exits non-zero if any trip fails.
"""
from __future__ import annotations

import argparse
import csv
import socket
import sys
import time
from pathlib import Path

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args()
    api = args.api.rstrip("/")

    t0 = time.perf_counter()
    h = requests.get(f"{api}/health", timeout=10).json()
    print(f"health: {h}  ({(time.perf_counter() - t0) * 1000:.0f} ms)")
    trips = requests.get(f"{api}/trips", timeout=30).json()["trips"]
    print(f"trips: {len(trips)}")

    rows, failures = [], 0
    for t in trips:
        tid = t["trip_id"]
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{api}/predict", json={"trip_id": tid}, timeout=args.timeout)
            r.raise_for_status()
            res = r.json()
            rt = time.perf_counter() - t0
            m = res.get("metrics", {})
            rows.append({"trip_id": tid, "n_points": res["n_points"], "model": res.get("model"),
                         "top1_hmm": m.get("top1_hmm"), "top1_clf": m.get("top1_clf"),
                         "route_f1": m.get("route_f1"), "continuity_hmm": m.get("continuity_hmm"),
                         "server_s": round(res["timings_ms"].get("server_total", 0) / 1000, 3),
                         "roundtrip_s": round(rt, 3), "response_kb": round(len(r.content) / 1024, 1),
                         "status": "ok"})
            print(f"  {tid}: ok  points={res['n_points']:>6}  top1={m.get('top1_hmm')}  "
                  f"server={rows[-1]['server_s']}s  roundtrip={rt:.2f}s")
        except Exception as e:
            failures += 1
            rows.append({"trip_id": tid, "status": f"FAIL {type(e).__name__}: {e}"})
            print(f"  {tid}: FAIL {e}")

    out = Path(__file__).resolve().parents[1] / "results" / "api" / f"latency_{socket.gethostname()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(trips) - failures}/{len(trips)} trips ok -> {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
