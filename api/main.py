"""M4 - PathSnap FastAPI backend (runs on Laptop 1).

Endpoints
    GET  /health                 -> service status, model name, number of available trips
    GET  /datasets               -> available trip sets (evaluator_test, student_test) and sizes
    GET  /trips?dataset=NAME     -> trips discovered at runtime in that trip set's folder
    POST /predict {"trip_id", "dataset"} -> full prediction pipeline + server-side scoring

Run (Laptop 1):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment
    PATHSNAP_EVAL_DIR    evaluator trips, one sub-folder per trip (default: data/evaluator_test)
    PATHSNAP_STUDENT_DIR development trips, second trip set (default: data/raw/student_test)
    PATHSNAP_ARTIFACTS  model/config folder (default: artifacts)
    PATHSNAP_STUB=1     force the nearest-arc stub predictor (testing before the model exists)

Integrity: the predictor only reads .track/.nodes/.arcs. The reference .route is read only by
the scoring step, and only aggregate metrics are returned - never the reference route itself.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # works with or without `pip install -e .`
sys.path.insert(0, str(ROOT))

from api.stub import StubPredictor, stub_score_trip  # noqa: E402

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "evaluator_test")).resolve()
STUDENT_DIR = Path(os.environ.get("PATHSNAP_STUDENT_DIR", ROOT / "data" / "raw" / "student_test")).resolve()
# selectable trip sets: the evaluator set (default, used for grading) and our development split
DATASETS = {"evaluator_test": EVAL_DIR, "student_test": STUDENT_DIR}
DEFAULT_DATASET = "evaluator_test"
ARTIFACTS = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts")).resolve()
FORCE_STUB = os.environ.get("PATHSNAP_STUB", "0") == "1"
REQUEST_LOG = ROOT / "results" / "api" / "requests.log"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("pathsnap.api")

STATE: dict = {}


def _load_predictor():
    """Real pipeline (M3) if available, otherwise the stub. Returns (predictor, scorer, name)."""
    if not FORCE_STUB:
        try:
            from pathsnap.pipeline import PathSnapPredictor  # M3
            from pathsnap.scoring import score_trip          # M3
            predictor = PathSnapPredictor.load(str(ARTIFACTS))
            name = "pathsnap"
            card = ARTIFACTS / "model_card.json"
            if card.exists():
                name = json.loads(card.read_text()).get("model", name)
            return predictor, score_trip, name
        except Exception as e:  # missing module/artifacts -> keep serving with the stub
            log.warning("Real predictor unavailable (%s: %s) -> using STUB predictor", type(e).__name__, e)
    return StubPredictor.load(str(ARTIFACTS)), stub_score_trip, StubPredictor.name


def _dataset_dir(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise HTTPException(status_code=404, detail=f"unknown dataset '{dataset}' (use one of {list(DATASETS)})")
    return DATASETS[dataset]


def _discover_trips(dataset: str = DEFAULT_DATASET) -> dict[str, Path]:
    """trip_id -> folder, for every sub-folder of the dataset folder that contains <id>.track (no hardcoding)."""
    trips = {}
    base = _dataset_dir(dataset)
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if d.is_dir() and (d / f"{d.name}.track").is_file():
                trips[d.name] = d
    return trips


def _trip_summary(tid: str, d: Path) -> dict:
    n, t_first, t_last = 0, None, None
    with open(d / f"{tid}.track") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                n += 1
                t = float(parts[2])
                t_first = t if t_first is None else t_first
                t_last = t
    return {"trip_id": tid, "n_points": n,
            "duration_s": round(t_last - t_first, 1) if n else 0.0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor, scorer, name = _load_predictor()
    STATE.update(predictor=predictor, scorer=scorer, model=name)
    REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    for ds, d in DATASETS.items():
        log.info("PathSnap API ready | model=%s | dataset=%s | dir=%s | trips=%d",
                 name, ds, d, len(_discover_trips(ds)))
    yield


app = FastAPI(title="PathSnap Map-Matching API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str = Field(..., examples=["00000000"])
    dataset: str = Field(DEFAULT_DATASET, examples=["evaluator_test", "student_test"])


@app.get("/health")
def health():
    counts = {ds: len(_discover_trips(ds)) for ds in DATASETS}
    return {"status": "ok", "model": STATE.get("model"), "n_trips": counts[DEFAULT_DATASET],
            "datasets": counts, "eval_dir": str(EVAL_DIR)}


@app.get("/datasets")
def datasets():
    return {"default": DEFAULT_DATASET,
            "datasets": [{"name": ds, "n_trips": len(_discover_trips(ds))} for ds in DATASETS]}


@app.get("/trips")
def trips(dataset: str = DEFAULT_DATASET):
    found = _discover_trips(dataset)
    return {"dataset": dataset, "trips": [_trip_summary(tid, d) for tid, d in found.items()]}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    # sync endpoint -> FastAPI runs it in a threadpool, so the server stays responsive
    found = _discover_trips(req.dataset)
    if req.trip_id not in found:  # whitelist check also prevents path traversal
        raise HTTPException(status_code=404, detail=f"unknown trip_id '{req.trip_id}' in dataset '{req.dataset}'")
    trip_dir = str(found[req.trip_id])
    client = request.client.host if request.client else "?"
    log.info("PREDICT start dataset=%s trip=%s from=%s", req.dataset, req.trip_id, client)

    t0 = time.perf_counter()
    try:
        pred = STATE["predictor"].predict_trip(trip_dir)
        t_pred = time.perf_counter()
        metrics = STATE["scorer"](trip_dir, pred)
    except Exception as e:
        log.exception("PREDICT FAILED trip=%s", req.trip_id)
        raise HTTPException(status_code=500, detail=f"prediction failed: {type(e).__name__}: {e}")
    t_end = time.perf_counter()

    pred_s, total_s = t_pred - t0, t_end - t0
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    log.info("PREDICT done  dataset=%s trip=%s from=%s n_points=%d model=%s top1_hmm=%s top1_clf=%s route_f1=%s "
             "continuity=%s predict=%.2fs total=%.2fs",
             req.dataset, req.trip_id, client, len(pred["gps_latlon"]), STATE["model"], fmt(metrics.get("top1_hmm")),
             fmt(metrics.get("top1_clf")), fmt(metrics.get("route_f1")), fmt(metrics.get("continuity_hmm")),
             pred_s, total_s)
    with open(REQUEST_LOG, "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": req.dataset, "trip_id": req.trip_id,
                             "client": client, "model": STATE["model"], "n_points": len(pred["gps_latlon"]),
                             "metrics": metrics, "predict_s": round(pred_s, 3),
                             "total_s": round(total_s, 3)}) + "\n")

    return {
        "trip_id": req.trip_id,
        "dataset": req.dataset,
        "model": STATE["model"],
        "n_points": len(pred["gps_latlon"]),
        "gps_latlon": pred["gps_latlon"],
        "route_latlon": pred["route_latlon"],
        "route_arcs": pred["route_arcs"],
        "obs_arc": pred["obs_arc"],
        "metrics": metrics,
        "timings_ms": pred.get("timings_ms", {}) | {"scoring": round((t_end - t_pred) * 1000, 1),
                                                    "server_total": round(total_s * 1000, 1)},
    }
