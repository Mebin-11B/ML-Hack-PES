# PathSnap — Two-Laptop Deployment (ZeroTier)

```
 Laptop 2 (UI)                           ZeroTier private network               Laptop 1 (Model + API)
 streamlit run ui/app.py   ── HTTP JSON ──►  10.x.x.x (virtual LAN)  ──►  uvicorn api.main:app :8000
 shows trips, map, metrics  ◄────────────────────────────────────────────  predictor + server-side scoring
```

- **Laptop 1** holds the trained model, the HMM/Viterbi code, the road networks and the trips; it runs the prediction pipeline and computes accuracy against the reference route **server-side**.
- **Laptop 2** runs only the UI. It sends a trip ID and displays the returned route + aggregate metrics. It never imports the prediction code and never receives reference routes.

## 1. One-time setup (both laptops)

```bash
git clone git@github.com:Mebin-11B/ML-Hack-PES.git && cd ML-Hack-PES
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. ZeroTier network

1. Sign in at <https://my.zerotier.com> → **Create A Network**. Keep **Access Control = Private** and IPv4 auto-assign on. Note the 16-character **Network ID** (never commit it).
2. Install ZeroTier One on both laptops from <https://www.zerotier.com/download/>.
   - Windows: run the MSI → tray icon → *Join New Network…* → paste the Network ID. If Windows asks whether to make the PC discoverable, answer **Yes** (Private network).
   - macOS: install the .pkg → menu-bar icon → *Join New Network…*; allow the system extension in System Settings → Privacy & Security.
   - Linux: `curl -s https://install.zerotier.com | sudo bash` then `sudo zerotier-cli join <NETWORK_ID>`.
3. In ZeroTier Central → **Members**, tick **Auth** for both laptops and name them `L1-API` and `L2-UI`. Optionally give L1 a fixed managed IP so the UI URL never changes.
4. Check: the tray/menu shows the network as **OK** with a managed IP. On Linux/macOS, `sudo zerotier-cli listnetworks` shows `OK PRIVATE <ip>`, and `sudo zerotier-cli peers` should list the other laptop as `DIRECT` (`RELAY` works but is slower and usually means the Wi-Fi blocks UDP).
5. **Allow the API port on Laptop 1:**
   - Windows (PowerShell as Administrator):
     ```powershell
     Get-NetConnectionProfile   # find the "ZeroTier One [...]" alias
     Set-NetConnectionProfile -InterfaceAlias "ZeroTier One [<id>]" -NetworkCategory Private
     New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any
     ```
     When Python first opens the port, allow it on **Private and Public** networks. Windows blocks ping by default, so test with HTTP rather than ping.
   - macOS: System Settings → Network → Firewall → allow incoming connections for Python.
   - Linux (only if ufw is active): `sudo ufw allow 8000/tcp`.
6. Network smoke test, before running the API: on L1 run `python -m http.server 8000 --bind 0.0.0.0`; on L2 open `http://<L1-ZT-IP>:8000`. If the page loads, ZeroTier and the firewall are working.

ZeroTier needs outbound **UDP 9993**. If the venue Wi-Fi blocks it or isolates clients, connect both laptops to the same phone hotspot. ZeroTier then finds a direct path, or the UI can use L1's hotspot LAN IP directly.

## 3. Run

**Laptop 1 (API):**
```bash
# evaluation trips: one sub-folder per trip, e.g. data/evaluator_test/<trip_id>/<trip_id>.track|.nodes|.arcs|.route
uvicorn api.main:app --host 0.0.0.0 --port 8000
# (use --host 0.0.0.0, not 127.0.0.1, otherwise Laptop 2 cannot connect)
# optional: PATHSNAP_EVAL_DIR=<folder>   PATHSNAP_STUB=1 (nearest-arc stub for testing)
```
The terminal prints one `PREDICT start` and one `PREDICT done …` line per request, giving the trip, the client IP, accuracy and latency. The same lines are appended as JSON to `results/api/requests.log`. Interactive docs are at `http://<L1>:8000/docs`.

**Laptop 2 (UI):**
```bash
streamlit run ui/app.py --server.port 8501
```
Enter `http://<L1-ZT-IP>:8000` in the sidebar. The badge turns green. Choose the trip set (**Evaluator test**, the default, or **Student test**, our development split), select a trip and press **Run map-matching**. The session history records which trip set each run used.

**Rehearsal (from Laptop 2):** `python api/smoke_test.py --api http://<L1-ZT-IP>:8000` runs every trip through the API and writes `results/api/latency_<host>.csv`.

## 4. API reference

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | – | `{"status":"ok","model":…,"n_trips":…}` |
| GET | `/datasets` | – | trip sets and their sizes: `evaluator_test` (default, `data/evaluator_test/`) and `student_test` (`data/raw/student_test/`) |
| GET | `/trips?dataset=…` | – | `{"trips":[{"trip_id","n_points","duration_s"}]}` (discovered from the folder at runtime) |
| POST | `/predict` | `{"trip_id":"…","dataset":"evaluator_test"}` | `trip_id, model, n_points, gps_latlon, route_latlon, route_arcs, obs_arc, metrics{top1_hmm, top1_clf, top1_hmm_undirected, route_precision, route_recall, route_f1, continuity_hmm, continuity_clf, n_breaks_hmm, n_valid_points}, timings_ms` |

Unknown trip IDs return 404; the check against the discovered trip list also blocks path traversal. If a trip has no `.route` file, every metric comes back as `null`.

## 5. Demo checklist
- [ ] Both laptops: same `git log -1`, venv active, charger plugged in, sleep disabled
- [ ] L1: evaluation trips in `data/evaluator_test/`, API running in a large-font terminal, the startup log shows the real model name (not `stub-nearest-arc`)
- [ ] L2: `/health` green over ZeroTier, 10 trips listed
- [ ] Hotspot fallback tested once
