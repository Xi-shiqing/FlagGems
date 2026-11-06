"""Verify actual registered forward/backward precision, preserving current RNE."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
sys.meta_path[:]=[x for x in sys.meta_path if '_flag_gems_editable' not in getattr(x,'__module__','')]
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from collections import Counter
import flag_gems

class RecordATen(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = Counter()
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if any(isinstance(x, torch.Tensor) and x.is_cuda for x in args):
            self.calls[str(func)] += 1
        return func(*args, **(kwargs or {}))

torch.set_num_threads(4)
generator=torch.Generator().manual_seed(20261106)
report={'source_metadata_note':'sources names exported Python functions; actual F.linear uses native composition when the PPU policy excludes linear. Per-case ATen calls record the executed forward/backward.', 'package':flag_gems.__file__,'seed':20261106,'cases':[],'sources':{}}
for name in ['mm','addmm','bmm','linear','softmax','softmax_backward']:
    fn=getattr(flag_gems,name,None)
    if fn is not None:
        path=Path(fn.__code__.co_filename)
        report['sources'][name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def rounded(x,tf32):
    if not tf32:return x.double()
    bits=x.contiguous().view(torch.int32).to(torch.int64)
    bits=(bits+4095+((bits>>13)&1))&-8192
    return bits.to(torch.int32).view(torch.float32).double()

def metrics(a,b):
    a=a.detach().cpu().double();b=b.double();assert a.shape==b.shape and torch.isfinite(a).all()
    d=(a-b).norm().item();n=b.norm().item()
    return {'difference_l2':d,'reference_l2':n,'relative_l2':d/max(n,1e-300),
            'max_abs':(a-b).abs().max().item()}

for name in ['mm','addmm','bmm','linear','softmax']:
    if name=='softmax':
        cpu=[torch.randn((600,600,8),generator=generator)]
        operation=lambda x:torch.softmax(x,dim=1)
    else:
        batch=(2,) if name=='bmm' else ()
        a=torch.randn((*batch,64,128),generator=generator)
        b=torch.randn((*batch,128,64),generator=generator)
        if name=='addmm':
            cpu=[torch.randn((64,),generator=generator),a,b]
            operation=lambda bias,x,w:torch.addmm(bias,x,w)
        elif name=='linear':
            cpu=[a,b.T.contiguous(),torch.randn((64,),generator=generator)]
            operation=F.linear
        else:
            cpu=[a,b];operation=torch.bmm if name=='bmm' else torch.mm
    gpu=[x.to('cuda').requires_grad_() for x in cpu]
    dy=torch.randn(operation(*cpu).shape,generator=generator);device_dy=dy.to('cuda')
    for precision in ['fp32','tf32']:
        tf32=precision=='tf32';torch.backends.cuda.matmul.allow_tf32=tf32
        if name=='softmax':
            reference=torch.softmax(cpu[0].double(),dim=1)
            references=[reference*(dy.double()-(reference*dy.double()).sum(1,keepdim=True))]
        else:
            left=cpu[1] if name=='addmm' else cpu[0]
            right=cpu[2] if name=='addmm' else (cpu[1].T if name=='linear' else cpu[1])
            qa,qb,qdy=rounded(left,tf32),rounded(right,tf32),rounded(dy,tf32)
            reference=qa@qb
            if name in ['addmm','linear']:reference=reference+cpu[0 if name=='addmm' else 2].double()
            da=qdy@qb.transpose(-1,-2);db=qa.transpose(-1,-2)@qdy
            references=([dy.double().sum(0),da,db] if name=='addmm' else
                        [da,db.T,dy.double().sum(0)] if name=='linear' else [da,db])
        row={'operator':name,'precision':precision,'shapes':[list(x.shape) for x in cpu]}
        try:
            with flag_gems.use_gems(exclude=[]):
                recorder=RecordATen()
                with recorder:
                    output=operation(*gpu)
                    gradients=torch.autograd.grad(output,gpu,device_dy)
                row['actual_ATen_calls']=dict(recorder.calls)
                row['forward']=metrics(output,reference)
                row['gradients']=[metrics(x,y) for x,y in zip(gradients,references)]
                assert row['forward']['relative_l2']<2e-5
                assert all(x['relative_l2']<2e-5 for x in row['gradients'])
                samples=[]
                with torch.no_grad():
                    for _ in range(3):operation(*gpu)
                    torch.cuda.synchronize()
                    for _ in range(7):
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(3):value=operation(*gpu)
                        end.record();end.synchronize();samples.append(start.elapsed_time(end)/3)
                row['forward_no_grad']=metrics(value,reference)
                assert row['forward_no_grad']['relative_l2']<2e-5
                row.update(status='PASS',device_ms=samples,median_ms=statistics.median(samples))
        except Exception as exc:
            row.update(status='FAIL',error=str(exc)[:2000])
        report['cases'].append(row)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        print('PRECISION_ROUTE',name,precision,row['status'],flush=True)
report['status']='PASS_ALL_CURRENT_PRECISION_ROUTES' if all(x['status']=='PASS' for x in report['cases']) else 'FAILED_PRECISION_ROUTE_CHECK'
Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
print(report['status'],flush=True)

assert report["status"]=="PASS_ALL_CURRENT_PRECISION_ROUTES", "Inspect per-route failure"
