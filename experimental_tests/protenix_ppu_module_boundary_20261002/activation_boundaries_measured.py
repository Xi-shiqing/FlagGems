"""Read-only hooks: full local errors plus non-reconstructible aggregate checksums.

No activation/weight/coordinate files are written. CPU activation references live
only during the first FP32/TF32 predictions. The 64 checksums each aggregate many
elements and estimate cross-device differences; they are not an accuracy gate.
"""
from collections import Counter
import hashlib
import numpy as np

CHUNK=262144
BUCKETS=64


def summaries(array):
    a=np.asarray(array).reshape(-1)
    total=square=0.0
    buckets=np.zeros(BUCKETS,dtype=np.float64)
    for start in range(0,a.size,CHUNK):
        x=a[start:start+CHUNK].astype(np.float64)
        assert np.isfinite(x).all()
        total+=float(x.sum());square+=float(np.dot(x,x))
        if a.size>=4096:
            # Fixed index hash; no RNG calls and no selected activation values.
            index=np.arange(start,start+x.size,dtype=np.uint64)
            h=index+np.uint64(0x9e3779b97f4a7c15)
            h=(h^(h>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
            h=(h^(h>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
            h=h^(h>>np.uint64(31))
            sign=1.0-2.0*((h>>np.uint64(63)).astype(np.float64))
            buckets+=np.bincount((h%BUCKETS).astype(np.int64),weights=x*sign,minlength=BUCKETS)
    return {'numel':int(a.size),'sum_fp64':total,'squared_norm_fp64':square,
            'rms':float(np.sqrt(square/max(1,a.size))),
            'aggregate_checksums':{f'bucket_{i:02d}':float(v) for i,v in enumerate(buckets)} if a.size>=4096 else None}


def error(array,reference):
    a,b=array.reshape(-1),reference.reshape(-1)
    assert a.shape==b.shape
    delta=norm=max_abs=0.0
    for start in range(0,a.size,CHUNK):
        x,y=[v[start:start+CHUNK].astype(np.float64) for v in (a,b)]
        d=x-y
        delta+=float(np.dot(d,d));norm+=float(np.dot(y,y))
        max_abs=max(max_abs,float(np.max(np.abs(d))))
    return {'relative_l2':float(np.sqrt(delta/max(norm,1e-60))),
            'max_abs':max_abs,'rms':float(np.sqrt(delta/max(1,a.size))),'finite':True}


class Boundaries:
    def __init__(self,model):
        import torch
        self.mode=None
        self.rows={}
        self.references={}
        self.arrays={}
        self.summary_cache={}
        self.calls=Counter()
        self.event=0
        self.linear_calls=0
        self.pending={}
        self.handles=[]
        self.selected=[]
        for name,module in model.named_modules():
            parts=name.split('.')
            block=len(parts)>1 and parts[-2]=='blocks' and parts[-1].isdigit()
            boundary=bool(name) and ('.' not in name or block or
                type(module).__name__ in {'AtomAttentionEncoder','AtomAttentionDecoder','DiffusionConditioning'})
            linear=isinstance(module,torch.nn.Linear)
            if not boundary and not linear:
                continue
            if boundary:self.selected.append({'module':name,'class':type(module).__name__})
            def before(mod,args,kwargs,name=name,boundary=boundary,linear=linear):
                if self.mode is None:return
                if linear:
                    self.linear_calls+=1
                    if not boundary and self.linear_calls>12:return
                self.calls[name]+=1
                self.event+=1
                row={'event':self.event,'module':name,'class':type(mod).__name__,
                     'call':self.calls[name],'kind':'boundary' if boundary else 'early_linear'}
                row['inputs']=self.describe((args,kwargs),name,self.calls[name],'input')
                self.pending.setdefault(name,[]).append(row)
            def after(mod,args,kwargs,output,name=name):
                if self.mode is None or not self.pending.get(name):return
                row=self.pending[name].pop()
                row['outputs']=self.describe(output,name,row['call'],'output')
                self.rows[self.mode].append(row)
            self.handles.append(module.register_forward_pre_hook(before,with_kwargs=True))
            self.handles.append(module.register_forward_hook(after,with_kwargs=True))

    def begin(self,mode):
        assert mode in ('fp32','tf32') and mode not in self.rows
        self.mode=mode;self.rows[mode]=[]
        self.calls.clear();self.pending.clear();self.event=self.linear_calls=0
        self.summary_cache={}

    def end(self):
        assert all(not rows for rows in self.pending.values())
        self.rows[self.mode].sort(key=lambda x:x['event'])
        self.mode=None
        self.summary_cache={}
        if 'tf32' in self.rows:
            self.references.clear();self.arrays.clear()

    def describe(self,value,name,call,side):
        import torch
        result={}
        def visit(v,path):
            if isinstance(v,torch.Tensor):
                if not v.is_floating_point() or v.numel()==0:return
                # Model protocol uses float32 storage. Hash the actual storage dtype.
                cpu=v.detach().cpu().contiguous()
                array=cpu.numpy()
                sha=hashlib.sha256(array.tobytes()).hexdigest()
                key=(name,call,side,path)
                summary=self.summary_cache.get(sha)
                if summary is None:
                    summary=summaries(array);self.summary_cache[sha]=summary
                record=dict(summary,shape=list(v.shape),dtype=str(v.dtype),sha256=sha)
                if self.mode=='fp32':
                    stored=self.arrays.setdefault((sha,array.shape,array.dtype.str),array)
                    self.references[key]=(sha,stored)
                elif key in self.references:
                    old_sha,reference=self.references[key]
                    if reference.shape!=array.shape:
                        record['vs_own_fp32']={'comparable':False,'reason':'input layout/shape changed across prediction calls','reference_shape':list(reference.shape)}
                        assert side=='input',(key,reference.shape,array.shape)
                        result[path]=record
                        return
                    record['vs_own_fp32']=({'relative_l2':0.0,'max_abs':0.0,'rms':0.0,'finite':True}
                        if sha==old_sha else error(array,reference))
                else:
                    raise AssertionError(('missing matched FP32 boundary',key))
                result[path]=record
            elif isinstance(v,dict):
                for k in sorted(v):visit(v[k],path+'.'+str(k))
            elif isinstance(v,(tuple,list)):
                for i,x in enumerate(v):visit(x,path+'.'+str(i))
        visit(value,'tensor')
        return result

    def close(self):
        for h in self.handles:h.remove()
        assert self.rows.keys()=={'fp32','tf32'}
        return {'status':'PASS','selected_boundaries':self.selected,'modes':self.rows,
                'scope':'First FP32/TF32 warmups only. Full same-device activation errors in memory; hashes, norms and 64 aggregate sums exported. Checksums estimate cross-device L2 and do not establish an acceptance gate.'}
