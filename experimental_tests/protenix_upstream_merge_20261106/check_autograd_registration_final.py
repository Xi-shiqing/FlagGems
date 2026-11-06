"""Compare real dispatcher autograd with independent native CPU gradients."""
import argparse
import json
from pathlib import Path
import sys
sys.meta_path[:]=[x for x in sys.meta_path if '_flag_gems_editable' not in getattr(x,'__module__','')]
import torch
import torch.nn.functional as F
import flag_gems

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
torch.backends.cuda.matmul.allow_tf32=False
generator=torch.Generator().manual_seed(20261106)
cpu=torch.rand((3,5,17),generator=generator)+.75
cases={
 'clone':lambda x:x.clone(),
 'clone_nonleaf':lambda x:(x*2).clone(),
 'cdist_self':lambda x:torch.cdist(x,x,p=2),
 'cdist_cross':lambda x:torch.cdist(x,x*1.1,p=2),
 'contiguous_transpose':lambda x:x.transpose(0,1).contiguous(),
 'pow2':lambda x:x.pow(2),
 'square':torch.square,
 'sqrt':torch.sqrt,
 'rsqrt':torch.rsqrt,
 'exp':torch.exp,
 'mean':lambda x:x.mean(1),
 'sum':lambda x:x.sum(1),
 'sigmoid':torch.sigmoid,
 'silu':F.silu,
 'gelu':F.gelu,
 'softmax':lambda x:torch.softmax(x,dim=1),
 'log_softmax':lambda x:torch.log_softmax(x,dim=1),
 'layer_norm':lambda x:F.layer_norm(x,(17,)),
 'vector_norm':lambda x:torch.linalg.vector_norm(x,dim=-1),
 'where':lambda x:torch.where(x>1,x,x*2),
 'reshape':lambda x:x.reshape(-1,17),
 'max_pool3d':lambda x:F.max_pool3d(x.reshape(1,1,3,5,17),(1,2,3)),
}
rows=[]
for name,fn in cases.items():
 reference_input=cpu.clone().double().requires_grad_();reference=fn(reference_input)
 upstream=torch.randn(reference.shape,generator=generator,dtype=torch.float32)
 reference_grad=torch.autograd.grad(reference,reference_input,upstream.double())[0]
 actual_input=cpu.to('cuda').requires_grad_();row={'operator':name}
 try:
  with flag_gems.use_gems(exclude=[]):
   actual=fn(actual_input);row['requires_grad']=actual.requires_grad;row['grad_fn']=type(actual.grad_fn).__name__
   assert actual.requires_grad,'Registered forward lost autograd'
   gradient=torch.autograd.grad(actual,actual_input,upstream.to('cuda'))[0]
  for key,a,b in [('forward',actual,reference),('gradient',gradient,reference_grad)]:
   a=a.detach().cpu().double();b=b.detach();n=b.norm().item();d=(a-b).norm().item()
   row[key]={'difference_l2':d,'reference_l2':n,'relative_l2':d/max(n,1e-300),'max_abs':(a-b).abs().max().item()}
   assert torch.isfinite(a).all()
   torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
  row['status']='PASS'
 except Exception as error:row.update(status='FAIL',error=str(error))
 rows.append(row);print('AUTOGRAD_ENTRY',name,row['status'],flush=True)
 Path(args.output).write_text(json.dumps({'package':flag_gems.__file__,'rows':rows},indent=2)+'\n')

assert len(rows)==len(cases) and all(row["status"]=="PASS" for row in rows), "Inspect failing registered VJP"
