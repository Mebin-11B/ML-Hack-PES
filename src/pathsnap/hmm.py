"""Candidate-state log-space Viterbi map matcher."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np

@dataclass
class HMMConfig:
    radius:float=50.; k_min:int=8; k_max:int=30; T:float=2.; beta:float=25.; G:float=60.; D0:float=200.; v_max:float=50.; backward_tol:float=15.
    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text())
        if 'gap_s' in data: data['G']=data.pop('gap_s')
        return cls(**data)
    def save(self,path):
        data=asdict(self); data['gap_s']=data.pop('G')
        Path(path).write_text(json.dumps(data,indent=2))

class ViterbiDecoder:
    def __init__(self,config=None,**kwargs):
        if isinstance(config,HMMConfig): self.cfg=config
        else: self.cfg=HMMConfig(**kwargs)
    def decode(self,trip,feat_df,probs,graph):
        n=len(trip.track); result=np.full(n,-1,dtype=np.int64)
        if not len(feat_df): return result,[]
        groups=[g for _,g in feat_df.groupby('obs_idx',sort=True)]
        if not groups:return result,[]
        arcsets=[g.arc_id.to_numpy(np.int64) for g in groups]
        ds=[g.dist_m.to_numpy(float) for g in groups]
        fracs=[g.proj_frac.to_numpy(float) for g in groups]
        p=np.asarray(probs,float)
        # Candidate rows are ordered with groups; normalize classifier mass per observation.
        em=[]; off=0
        for g in groups:
            q=np.maximum(p[off:off+len(g)],0); off+=len(g)
            q=(q+1e-9)/(q.sum()+1e-9*len(q)); em.append(np.log(q)/max(self.cfg.T,1e-6))
        xy=np.column_stack(graph.projector.fwd(trip.track[:,0],trip.track[:,1]))
        obs=[int(g.obs_idx.iloc[0]) for g in groups]; back=[None]*len(groups)
        score=em[0].copy()
        for i in range(1,len(groups)):
            pi,ci=arcsets[i-1],arcsets[i]; prevobs,curobs=obs[i-1],obs[i]
            dt=float(trip.track[curobs,2]-trip.track[prevobs,2]); gc=float(np.linalg.norm(xy[curobs]-xy[prevobs]))
            trans=np.full((len(pi),len(ci)),-np.inf)
            if dt<=self.cfg.G:
                for u,a in enumerate(pi):
                    fa,ta=graph.arcs[a]; sa=float(fracs[i-1][u]*graph.arc_len[a])
                    drow, _=graph.distances(ta, max(gc+self.cfg.D0+self.cfg.v_max*dt, 1000))
                    for v,b in enumerate(ci):
                        fb,tb=graph.arcs[b]; sb=float(fracs[i][v]*graph.arc_len[b])
                        if a==b:
                            delta=sb-sa
                            if delta < -self.cfg.backward_tol: continue
                            dn=max(0.,delta)
                        else:
                            sp=drow[int(fb)]
                            if not np.isfinite(sp): continue
                            dn=max(0.,graph.arc_len[a]-sa)+sp+sb
                        if dn<=gc+max(self.cfg.D0,self.cfg.v_max*dt): trans[u,v]=-abs(dn-gc)/self.cfg.beta
            mat=score[:,None]+trans
            bp=np.argmax(mat,axis=0); new=mat[bp,np.arange(len(ci))]+em[i]
            if not np.isfinite(new).any(): score=em[i].copy(); back[i]=np.full(len(ci),-1,np.int32)
            else: score=new; back[i]=bp.astype(np.int32)
        j=int(np.argmax(score)); states=[0]*len(groups)
        for i in range(len(groups)-1,-1,-1):
            states[i]=j
            if i and back[i] is not None and back[i][j]>=0: j=int(back[i][j])
            elif i: j=int(np.argmax(em[i-1]))
        for o,a,s in zip(obs,arcsets,states): result[o]=a[s]
        return result,arcsets
