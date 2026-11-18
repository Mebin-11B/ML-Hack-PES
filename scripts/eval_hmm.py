"""Compare geometry, classifier and HMM on a permitted split; write auditable per-trip rows."""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np,pandas as pd
import folium
from pathsnap.io import list_trips,load_trip
from pathsnap.pipeline import PathSnapPredictor
from pathsnap.labels import align_route_to_track
from pathsnap.metrics import top1_accuracy,top1_undirected,route_continuity,route_prf
from pathsnap.routing import RoadGraph
from pathsnap.candidates import RoadIndex,generate_candidates

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--split',default='student_test'); ap.add_argument('--artifacts',default='artifacts'); ap.add_argument('--out',default='results/m3/clf_vs_hmm.csv'); a=ap.parse_args()
    predictor=PathSnapPredictor.load(a.artifacts); rows=[]; map_cases=[]
    for path in list_trips(Path('data/raw')/a.split):
        trip=load_trip(path,True); t0=time.perf_counter(); pred=predictor.predict_trip(path); elapsed=time.perf_counter()-t0
        gt,dist=align_route_to_track(trip); valid=dist<=30; graph=RoadGraph(trip); ri=RoadIndex(trip,graph.projector)
        cand=generate_candidates(trip,graph.projector,ri,predictor.config.radius,predictor.config.k_min,predictor.config.k_max)
        nearest=cand.sort_values(['obs_idx','dist_m'],kind='stable').drop_duplicates('obs_idx').set_index('obs_idx').arc_id.reindex(range(len(trip.track))).to_numpy()
        methods=[('Nearest arc',nearest,graph.gap_fill_route(nearest)),('Classifier only',np.asarray(pred['obs_arc_clf']),graph.gap_fill_route(pred['obs_arc_clf'])),('Classifier + HMM',np.asarray(pred['obs_arc']),pred['route_arcs'])]
        for name,obs,route in methods:
            con=route_continuity(trip.arcs,obs); pr=route_prf(route,trip.route,graph.arc_len)
            rows.append({'trip_id':trip.trip_id,'method':name,'Top-1':top1_accuracy(obs,gt,valid),'undirected Top-1':top1_undirected(obs,gt,valid,graph.twin_of),'route F1':pr['f1'],'continuity':con['continuity'],'breaks':con['n_breaks'],'runtime_s':elapsed,'n_valid_points':int(valid.sum())})
        map_cases.append((top1_accuracy(pred['obs_arc'],gt,valid)-top1_accuracy(pred['obs_arc_clf'],gt,valid),trip,pred))
        print(f"{trip.trip_id}: HMM top1={rows[-1]['Top-1']:.3f}, F1={rows[-1]['route F1']:.3f}, {elapsed:.2f}s")
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); df=pd.DataFrame(rows); df.to_csv(out,index=False)
    summary=df.groupby('method')[['Top-1','undirected Top-1','route F1','continuity','breaks']].mean().reset_index()
    summary.to_csv(out.parent/'clf_vs_hmm_summary.csv',index=False)
    print('Mean by method:'); print(df.groupby('method')[['Top-1','undirected Top-1','route F1','continuity','breaks','runtime_s']].mean().to_string())
    # Choose map examples from measured classifier-to-HMM changes, not a hardcoded trip.
    for tag,case in [('helps',max(map_cases,key=lambda x:x[0])),('fails',min(map_cases,key=lambda x:x[0]))]:
        _,trip,pred=case; graph=RoadGraph(trip); refs=graph.route_to_latlon(trip.route) if hasattr(graph,'route_to_latlon') else []
        if not refs:
            refs=[]
            for arc in trip.route:
                for node in trip.arcs[int(arc)]:
                    pt=[float(trip.nodes[node,1]),float(trip.nodes[node,0])]
                    if not refs or refs[-1]!=pt: refs.append(pt)
        pts=pred['gps_latlon']+refs+pred['route_latlon']; center=[float(np.mean([p[0] for p in pts])),float(np.mean([p[1] for p in pts]))]
        m=folium.Map(location=center,zoom_start=14); folium.PolyLine(pred['gps_latlon'],color='blue',weight=2,name='GPS').add_to(m); folium.PolyLine(refs,color='green',weight=5,name='Reference').add_to(m); folium.PolyLine(pred['route_latlon'],color='red',weight=4,name='HMM').add_to(m)
        clf=[]
        for arc in graph.gap_fill_route(pred['obs_arc_clf']):
            for node in trip.arcs[arc]:
                pt=[float(trip.nodes[node,1]),float(trip.nodes[node,0])]
                if not clf or clf[-1]!=pt:clf.append(pt)
        folium.PolyLine(clf,color='orange',weight=3,name='Classifier').add_to(m); folium.LayerControl().add_to(m)
        dest=Path('results/maps')/f'hmm_{tag}_{trip.trip_id}.html'; dest.parent.mkdir(parents=True,exist_ok=True); m.save(dest)
if __name__=='__main__': main()
