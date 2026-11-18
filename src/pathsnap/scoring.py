"""Serve-side evaluation. This is the only prediction service module that reads .route."""
from pathlib import Path
import numpy as np
from .io import load_trip
from .labels import align_route_to_track
from .metrics import top1_accuracy,top1_undirected,route_continuity,route_prf
from .routing import RoadGraph

KEYS=['top1_hmm','top1_clf','top1_hmm_undirected','route_precision','route_recall','route_f1','continuity_hmm','continuity_clf','n_breaks_hmm','n_valid_points']
def score_trip(trip_dir,pred):
    if not list(Path(trip_dir).glob('*.route')): return {k:None for k in KEYS}
    trip=load_trip(trip_dir,with_route=True); gt,dist=align_route_to_track(trip); valid=dist<=30
    graph=RoadGraph(trip); h=route_continuity(trip.arcs,pred['obs_arc']); c=route_continuity(trip.arcs,pred['obs_arc_clf'])
    pr=route_prf(pred['route_arcs'],trip.route,graph.arc_len)
    return {'top1_hmm':top1_accuracy(pred['obs_arc'],gt,valid),'top1_clf':top1_accuracy(pred['obs_arc_clf'],gt,valid),'top1_hmm_undirected':top1_undirected(pred['obs_arc'],gt,valid,graph.twin_of),'route_precision':pr['precision'],'route_recall':pr['recall'],'route_f1':pr['f1'],'continuity_hmm':h['continuity'],'continuity_clf':c['continuity'],'n_breaks_hmm':h['n_breaks'],'n_valid_points':int(valid.sum())}
