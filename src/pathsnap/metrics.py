"""Shared metric definitions."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,log_loss,confusion_matrix

def top1_accuracy(pred_arc,gt_arc,valid):
    p,g,v=np.asarray(pred_arc),np.asarray(gt_arc),np.asarray(valid,bool); return float(np.mean(p[v]==g[v])) if v.any() else float('nan')
def top1_undirected(pred_arc,gt_arc,valid,twin_of):
    p,g,v=np.asarray(pred_arc),np.asarray(gt_arc),np.asarray(valid,bool); tw=np.asarray(twin_of(g)); return float(np.mean((p[v]==g[v])|(p[v]==tw[v]))) if v.any() else float('nan')
def candidate_miss_rate(cands_df):
    if 'gt_arc' not in cands_df:return float('nan')
    misses=[]
    for _,g in cands_df.groupby('obs_idx',sort=False):
        gt=int(g.gt_arc.iloc[0]); misses.append(not np.any(g.arc_id.to_numpy()==gt))
    return float(np.mean(misses)) if misses else float('nan')
def route_continuity(arcs,seq):
    s=[]
    for a in seq:
        a=int(a)
        if a>=0 and (not s or s[-1]!=a):s.append(a)
    if len(s)<2:return {'continuity':1.0,'n_breaks':0}
    ok=np.asarray(arcs)[np.asarray(s[:-1]),1]==np.asarray(arcs)[np.asarray(s[1:]),0]
    return {'continuity':float(ok.mean()),'n_breaks':int((~ok).sum())}
def route_prf(pred_route_arcs,ref_route_arcs,arc_len=None):
    p=set(map(int,pred_route_arcs)); r=set(map(int,ref_route_arcs)); tp=p&r
    prec=len(tp)/len(p) if p else 0.; rec=len(tp)/len(r) if r else 0.; out={'precision':prec,'recall':rec,'f1':2*prec*rec/(prec+rec) if prec+rec else 0.}
    if arc_len is not None:
        w=np.asarray(arc_len); wp=sum(w[x] for x in tp); out.update(weighted_precision=wp/sum(w[x] for x in p) if p else 0.,weighted_recall=wp/sum(w[x] for x in r) if r else 0.)
    return out
def pair_metrics(y_true,p):
    y=np.asarray(y_true); pred=np.asarray(p)>=.5; pr,re,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0); _,_,mf,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
    return {'acc':float(accuracy_score(y,pred)),'precision':float(pr),'recall':float(re),'f1':float(f1),'macro_f1':float(mf),'logloss':float(log_loss(y,np.clip(p,1e-9,1-1e-9))),'confusion':confusion_matrix(y,pred).tolist()}
