"""End-to-end predictor used by the FastAPI service; deliberately never loads .route."""
from __future__ import annotations
import json, logging, time
from pathlib import Path
import joblib, numpy as np
from .io import load_trip
from .geo import LocalProjector
from .candidates import RoadIndex,generate_candidates
from .features import add_features
from .hmm import HMMConfig,ViterbiDecoder
from .routing import RoadGraph

log=logging.getLogger(__name__)
class PathSnapPredictor:
    @classmethod
    def load(cls,artifacts_dir='artifacts'):
        p=Path(artifacts_dir); obj=cls(); obj.artifacts_dir=p
        obj.feature_list=[x['name'] for x in json.loads((p/'feature_list.json').read_text())]
        obj.model=joblib.load(p/'model.joblib') if (p/'model.joblib').exists() else None
        obj.config=HMMConfig.load(p/'hmm_config.json') if (p/'hmm_config.json').exists() else HMMConfig()
        log.info('PathSnap emissions: %s', 'classifier' if obj.model is not None else 'distance fallback')
        return obj
    def predict_trip(self,trip_dir):
        t0=time.perf_counter(); trip=load_trip(trip_dir,with_route=False); t1=time.perf_counter()
        proj=LocalProjector.from_trip(trip); ri=RoadIndex(trip,proj); graph=RoadGraph(trip,proj)
        c=generate_candidates(trip,proj,ri,self.config.radius,self.config.k_min,self.config.k_max); t2=time.perf_counter()
        f=add_features(c,trip,proj,ri); t3=time.perf_counter()
        if self.model is not None:
            probs=self.model.predict_proba(f[self.feature_list])[:,1]
        else: probs=np.exp(-.5*(f.dist_m.to_numpy()/10.)**2)
        # Classifier-only prediction is the highest probability within each observation.
        f=f.copy(); f['_p']=probs; idx=f.groupby('obs_idx',sort=True)['_p'].idxmax(); clf=np.full(len(trip.track),-1,np.int64); clf[f.loc[idx,'obs_idx'].to_numpy()]=f.loc[idx,'arc_id'].to_numpy()
        t4=time.perf_counter(); hmm=ViterbiDecoder(self.config); obs,_=hmm.decode(trip,f,probs,graph); route=graph.gap_fill_route(obs); t5=time.perf_counter()
        poly=[]
        for a in route:
            for node in trip.arcs[a]:
                lon,lat=trip.nodes[node]
                xy=[float(lat),float(lon)]
                if not poly or poly[-1]!=xy:poly.append(xy)
        ms=lambda a,b:round((b-a)*1000,1)
        return {'trip_id':trip.trip_id,'obs_arc':obs.tolist(),'obs_arc_clf':clf.tolist(),'route_arcs':route,'route_latlon':poly,'gps_latlon':np.c_[trip.track[:,1],trip.track[:,0]].round(7).tolist(),'timings_ms':{'load':ms(t0,t1),'candidates':ms(t1,t2),'features':ms(t2,t3),'classifier':ms(t3,t4),'viterbi':ms(t4,t5),'total':ms(t0,t5)}}
