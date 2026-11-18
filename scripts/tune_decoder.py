"""Small decoder grid on train trips; separately reports confirmation on student_test."""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np,pandas as pd
from pathsnap.io import list_trips,load_trip
from pathsnap.geo import LocalProjector
from pathsnap.candidates import RoadIndex,generate_candidates
from pathsnap.features import add_features
from pathsnap.pipeline import PathSnapPredictor
from pathsnap.hmm import HMMConfig,ViterbiDecoder
from pathsnap.routing import RoadGraph
from pathsnap.labels import align_route_to_track
from pathsnap.metrics import top1_accuracy,route_continuity,route_prf

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-dir',default='data/raw/train'); ap.add_argument('--test-dir',default='data/raw/student_test'); ap.add_argument('--artifacts',default='artifacts'); ap.add_argument('--n-trips',type=int,default=3); ap.add_argument('--out',default='results/m3/decoder_tuning.csv'); a=ap.parse_args()
    predictor=PathSnapPredictor.load(a.artifacts); grid=[(T,b,g) for T in (1.,2.) for b in (15.,25.) for g in (30.,60.)]; records=[]
    def prep(paths,split):
        data=[]
        for path in paths:
            trip=load_trip(path,True); proj=LocalProjector.from_trip(trip); ri=RoadIndex(trip,proj); graph=RoadGraph(trip,proj)
            c=generate_candidates(trip,proj,ri,predictor.config.radius,predictor.config.k_min,predictor.config.k_max); f=add_features(c,trip,proj,ri)
            probs=predictor.model.predict_proba(f[predictor.feature_list])[:,1]; gt,dist=align_route_to_track(trip)
            data.append((trip,f,probs,gt,dist,graph))
        for T,b,g in grid:
            scores=[]; starts=time.perf_counter()
            cfg=HMMConfig(**(predictor.config.__dict__|{'T':T,'beta':b,'G':g})); decoder=ViterbiDecoder(cfg)
            for trip,f,p,gt,dist,graph in data:
                obs,_=decoder.decode(trip,f,p,graph); valid=dist<=30; pr=route_prf(graph.gap_fill_route(obs),trip.route,graph.arc_len); con=route_continuity(trip.arcs,obs)
                scores.append((top1_accuracy(obs,gt,valid),pr['f1'],con['continuity'],con['n_breaks']))
            records.append({'split':split,'T':T,'beta':b,'gap_s':g,'mean_top1':float(np.nanmean([s[0] for s in scores])),'mean_route_f1':float(np.mean([s[1] for s in scores])),'mean_continuity':float(np.mean([s[2] for s in scores])),'mean_breaks':float(np.mean([s[3] for s in scores])),'runtime_s':time.perf_counter()-starts,'n_trips':len(data)})
    train_paths=list_trips(a.train_dir)[:a.n_trips]; prep(train_paths,'train')
    train=pd.DataFrame(records); best=train.sort_values(['mean_top1','mean_route_f1'],ascending=False).iloc[0]
    # Confirmation only: evaluate the train-selected configuration on the whole student_test split.
    test_paths=list_trips(a.test_dir); test_tripdata=[]
    for path in test_paths:
        trip=load_trip(path,True); proj=LocalProjector.from_trip(trip); ri=RoadIndex(trip,proj); graph=RoadGraph(trip,proj); c=generate_candidates(trip,proj,ri,predictor.config.radius,predictor.config.k_min,predictor.config.k_max); f=add_features(c,trip,proj,ri); p=predictor.model.predict_proba(f[predictor.feature_list])[:,1]; gt,d=align_route_to_track(trip); test_tripdata.append((trip,f,p,gt,d,graph))
    cfg=HMMConfig(**(predictor.config.__dict__|{'T':float(best['T']),'beta':float(best['beta']),'G':float(best['gap_s'])})); dec=ViterbiDecoder(cfg); vals=[]
    for trip,f,p,gt,d,graph in test_tripdata:
        obs,_=dec.decode(trip,f,p,graph); vals.append((top1_accuracy(obs,gt,d<=30),route_prf(graph.gap_fill_route(obs),trip.route,graph.arc_len)['f1'],route_continuity(trip.arcs,obs)['continuity']))
    records.append({'split':'student_test_confirmation','T':cfg.T,'beta':cfg.beta,'gap_s':cfg.G,'mean_top1':float(np.nanmean([v[0] for v in vals])),'mean_route_f1':float(np.mean([v[1] for v in vals])),'mean_continuity':float(np.mean([v[2] for v in vals])),'mean_breaks':np.nan,'runtime_s':np.nan,'n_trips':len(vals)})
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(records).to_csv(out,index=False); print(pd.DataFrame(records).to_string(index=False))
    HMMConfig(**(predictor.config.__dict__|{'T':cfg.T,'beta':cfg.beta,'G':cfg.G})).save(Path(a.artifacts)/'hmm_config.json')
if __name__=='__main__':main()
