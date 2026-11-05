"""Check the actual registered ATen backward from a one-file isolated package."""
import argparse
import importlib
import json
import os
from pathlib import Path

import flag_gems
import torch
from analyze_ln_abi_20261106 import formula, literal, metrics, sha

parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,required=True);args,_=parser.parse_known_args()
module=importlib.import_module('flag_gems.ops.layernorm')
expected=Path('/mnt/cpfs/xishiqing/ppu_full_acceptance_20261005/precision_residual_20261019/gems_ln_candidate_v3_20261106/flag_gems/ops/layernorm.py')
assert Path(module.__file__).resolve()==expected
manifest=json.loads(expected.parents[3].joinpath('ln_candidate_package_v3_20261106.json').read_text())
assert sha(expected)==manifest['candidate_sha256']
calls=[];original=module._fused_large_row_backward
def observed(*values):
    result=original(*values);calls.append({'M':values[-2],'N':values[-1],'fused_selected':result is not None});return result
module._fused_large_row_backward=observed
torch.set_num_threads(8)
state={'status':'RUNNING','package_manifest':manifest,'actual_layernorm_source':str(expected),'actual_sha256':sha(expected),'torch':torch.__version__,'cases':[],'new_H100_execution':False}
generator=torch.Generator().manual_seed(20261106)
with flag_gems.use_gems(include=['layer_norm_backward']):
    for shape,norm in [((2,3,7),(7,)),((2,3,4,5),(4,5)),((5,),(5,)),((4097,65),(65,)),((4097,384),(384,)),((4097,2048),(2048,))]:
        xx=torch.randn(shape,generator=generator);gg=torch.randn(shape,generator=generator)
        w=torch.randn(norm,generator=generator).cuda();b=torch.randn(norm,generator=generator).cuda()
        x=xx.cuda().requires_grad_();w.requires_grad_();b.requires_grad_()
        output,mean,rstd=torch.ops.aten.native_layer_norm.default(x,norm,w,b,1e-5)
        begin=len(calls)
        actual=torch.autograd.grad(output,(x,w,b),gg.cuda())
        assert len(calls)==begin+1
        expected_values=formula(gg,xx,norm,mean.detach().cpu(),rstd.detach().cpu(),w.detach().cpu(),b.detach().cpu())
        errors=[]
        for i,(a,e) in enumerate(zip(actual,expected_values)):
            value=metrics(a.cpu(),e.reshape(a.shape))
            assert value['absolute_L2']<=2e-5*(value['reference_squared']**.5)+2e-10,value
            errors.append(value)
        state['cases'].append({'shape':shape,'normalized_shape':norm,'errors':errors,'registered_backward_observation':calls[-1]})
        (args.output_root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
with flag_gems.use_gems(include=['layer_norm_backward']):
    for shape,norm in [((0,5),(5,)),((2,0),(0,))]:
        x=torch.empty(shape,dtype=torch.float32,requires_grad=True)
        w=torch.ones(norm,requires_grad=True);b=torch.zeros(norm,requires_grad=True)
        y,m,r=torch.ops.aten.native_layer_norm.default(x,norm,w,b,1e-5)
        dy=torch.empty_like(y);reference=torch.autograd.grad(y,(x,w,b),dy)
        actual=torch.ops.aten.native_layer_norm_backward.default(dy.cuda(),x.detach().cuda(),norm,m.cuda(),r.cuda(),w.detach().cuda(),b.detach().cuda(),(True,True,True))
        assert literal(actual,reference)
        state['cases'].append({'shape':shape,'normalized_shape':norm,'empty_against_CPU_native_autograd':'PASS'})
assert any(c['fused_selected'] for c in calls)
state['status']='PASS_REGISTERED_NATIVE_LAYERNORM_AUTOGRAD';(args.output_root/'result.json').write_text(json.dumps(state,indent=2)+'\n');print(state['status'],len(state['cases']),flush=True)
