"""Cross-device, deterministic full-output mean regression and backward checks."""
import argparse,contextlib,hashlib,importlib,json,statistics,sys,time
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--device',choices=['ppu','h100'],required=True);p.add_argument('--out',required=True);args=p.parse_args()
out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
ctx=contextlib.nullcontext;mod=None;old=None
if args.device=='ppu':
 import flag_gems
 mod=importlib.import_module('flag_gems.ops.mean')
 spec=importlib.util.spec_from_file_location('mean_before_fix',Path(__file__).with_name('mean_before.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 ctx=lambda:flag_gems.use_gems(exclude=[])
rows=[];shapes=[]
for n in [1,31,32,33,127,128,129,255,256,257,511,512,513,1023,1024,1025,1972,1973,1974,1975,2047,2048,2049,2529,3892,4095,4096,4097,5222,5370,8193,9142]:shapes.append((1,n,0))
for m in [2,3,4]:
 for n in [1023,1024,1025,1973,2047,2048,2049,4097,8193]:shapes.append((m,n,0))
for off in [1,2,3]:
 for m,n in [(1,1973),(2,1973),(3,2049),(1,5370)]:shapes.append((m,n,off))
for index,(m,n,offset) in enumerate(shapes):
 for kind in ['ones','ramp','random']:
  rng=np.random.Generator(np.random.PCG64(20260920+index))
  a=np.ones((m,n),np.float32) if kind=='ones' else (np.arange(m*n,dtype=np.float32).reshape(m,n)%2048-1024)/1024 if kind=='ramp' else rng.integers(-32768,32768,size=(m,n),dtype=np.int32).astype(np.float32)/32768
  storage=torch.empty(m*n+offset,device='cuda',dtype=torch.float32);x=storage[offset:].reshape(m,n);x.copy_(torch.from_numpy(a));reference=a.astype(np.float64).mean(-1)
  with ctx():
   y=torch.mean(x,dim=-1);keep=torch.mean(x,dim=-1,keepdim=True);flat=torch.mean(x)
  values=y.cpu().numpy();f=flat.item();k=keep.cpu().numpy()
  err=float(np.max(np.abs(values-reference)));flaterr=abs(f-float(a.astype(np.float64).mean()))
  assert err<=2e-7 and flaterr<=2e-7,(m,n,offset,kind,err,flaterr)
  assert np.array_equal(values,k.reshape(-1))
  row={'M':m,'N':n,'offset':offset,'kind':kind,'input_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'values':values.tolist(),'flat':f,'max_abs_fp64':err,'flat_abs_fp64':flaterr}
  if old and offset==0:
   with flag_gems.use_gems(exclude=[]):previous=old.mean_dim(x,dim=-1)
   prior=previous.cpu().numpy();width=mod._h100_inner_block_width(m,n)
   row['block_x']=width;row['old_values']=prior.tolist();row['old_new_bitwise']=bool(np.array_equal(prior.view(np.uint32),values.view(np.uint32)))
   if width!=256:assert row['old_new_bitwise'],('unaffected changed',row)
  rows.append(row)
# Exercise backward at the failing size, and the non-contiguous input route.
for transposed in [False,True]:
 a=np.arange(3*1973,dtype=np.float32).reshape(3,1973)/8192
 x=torch.from_numpy(a).cuda();x=x.T if transposed else x;x.requires_grad_(True)
 with ctx():
  y=x.mean(-1);y.sum().backward()
 ref=x.detach().cpu().double().mean(-1).numpy();actual=y.detach().cpu().numpy();grad=x.grad.cpu().numpy()
 assert np.max(np.abs(actual-ref))<2e-7
 assert np.all(grad==np.float32(1/x.shape[-1]))
 rows.append({'special':'backward_transposed' if transposed else 'backward','values':actual.tolist(),'gradient_sha256':hashlib.sha256(grad.tobytes()).hexdigest(),'max_abs_fp64':float(np.max(np.abs(actual-ref)))})
timings=[]
for m,n in [(1,1973),(2,1973),(1,5370)]:
 x=torch.ones((m,n),device='cuda')
 for label,fn in [('current',lambda:torch.mean(x,dim=-1))]+([('before',lambda:old.mean_dim(x,dim=-1))] if old else []):
  with ctx():
   for _ in range(3):fn()
   torch.cuda.synchronize();times=[]
   for _ in range(15):
    torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();times.append((time.perf_counter()-t)*1e6)
  timings.append({'M':m,'N':n,'mode':label,'median_us':statistics.median(times),'us':times})
result={'device':args.device,'passed':True,'cases':len(rows),'rows':rows,'timings':timings,'mean_source':getattr(mod,'__file__',None),'mean_sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() if mod else None,'native_h100':args.device=='h100' and 'flag_gems' not in sys.modules}
out.write_text(json.dumps(result,indent=2)+'\n');print('MEAN_REGRESSION_PASS',args.device,len(rows),flush=True)
