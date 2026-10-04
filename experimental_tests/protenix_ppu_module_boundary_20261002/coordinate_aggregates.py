"""Aggregate atom-coordinate checksums for approximate ranking, not RMSD gates."""
import numpy as np


def coordinate_aggregates(value):
    x=value.detach().cpu().numpy().astype(np.float64).reshape(-1,3)
    assert np.isfinite(x).all()
    n=len(x)
    if n<1024:
        return {'available':False,'reason':'too few atoms for aggregate-only diagnostics'}
    x=x-x.mean(0)
    index=np.arange(n,dtype=np.uint64)
    h=index+np.uint64(0x9e3779b97f4a7c15)
    h=(h^(h>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    h=(h^(h>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    h=h^(h>>np.uint64(31))
    buckets=(h%64).astype(np.int64)
    sign=1.0-2.0*((h>>np.uint64(63)).astype(np.float64))
    result={'available':True,'atom_count':n,
            'scope':'64 signed aggregate sums per coordinate axis over centered atoms; not atom positions. Approximate aligned RMSD ranking only, never an acceptance gate.'}
    for axis in range(3):
        sums=np.bincount(buckets,weights=x[:,axis]*sign,minlength=64)
        result['axis_'+str(axis)]={f'bucket_{i:02d}':float(v) for i,v in enumerate(sums)}
    return result


def estimated_rmsd(a,b):
    if not a['available'] or not b['available']:return None
    assert a['atom_count']==b['atom_count']
    p,q=[np.array([list(v['axis_'+str(i)].values()) for i in range(3)]).T for v in (a,b)]
    u,_,vh=np.linalg.svd(p.T@q)
    correction=np.eye(3);correction[-1,-1]=np.sign(np.linalg.det(u@vh))
    d=p@u@correction@vh-q
    return float(np.sqrt(np.square(d).sum()/a['atom_count']))
