"""Directed road graph utilities and bounded shortest paths."""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from .geo import LocalProjector

class RoadGraph:
    def __init__(self, trip, projector=None):
        self.trip = trip
        self.projector = projector or LocalProjector.from_trip(trip)
        xy = np.column_stack(self.projector.fwd(trip.nodes[:,0], trip.nodes[:,1]))
        self.node_xy = xy; self.arcs = trip.arcs; self.n_nodes=len(xy)
        a,b=self.arcs.T; self.arc_len=np.linalg.norm(xy[b]-xy[a],axis=1)
        self.G=csr_matrix((self.arc_len,(a,b)),shape=(self.n_nodes,self.n_nodes))
        self.cache={}
        self._lookup={}
        for i,(u,v) in enumerate(self.arcs): self._lookup.setdefault((int(u),int(v)),i)
    def distances(self, source, limit=np.inf):
        source=int(source)
        if source not in self.cache or self.cache[source][0] < limit:
            d,p=dijkstra(self.G,directed=True,indices=source,return_predecessors=True,limit=limit)
            self.cache[source]=(limit,d,p)
        return self.cache[source][1:]
    def shortest_path(self, source, target, limit=np.inf):
        d,p=self.distances(source,limit)
        if not np.isfinite(d[int(target)]): return [],np.inf
        nodes=[int(target)]; cur=int(target)
        while cur != int(source):
            cur=int(p[cur])
            if cur < 0: return [],np.inf
            nodes.append(cur)
        nodes.reverse(); ids=[self._lookup.get((u,v),-1) for u,v in zip(nodes[:-1],nodes[1:])]
        return ids,float(d[int(target)])
    def twin_of(self, arc_ids):
        lut={(int(a),int(b)):i for i,(a,b) in enumerate(self.arcs)}
        return np.array([lut.get((int(self.arcs[i,1]),int(self.arcs[i,0])),-1) for i in np.asarray(arc_ids).ravel()]).reshape(np.shape(arc_ids))
    def gap_fill_route(self, seq):
        seq=[int(x) for x in seq if int(x)>=0]
        out=[]
        for a in seq:
            if out and a==out[-1]: continue
            if out and self.arcs[out[-1],1] != self.arcs[a,0]:
                mid,_=self.shortest_path(self.arcs[out[-1],1],self.arcs[a,0])
                out.extend(x for x in mid if x>=0)
            out.append(a)
        return out
    def route_to_latlon(self, route):
        pts=[]
        for a in route:
            for node in self.arcs[int(a)]:
                lon,lat=self.trip.nodes[node]
                p=[float(lat),float(lon)]
                if not pts or pts[-1]!=p: pts.append(p)
        return pts
