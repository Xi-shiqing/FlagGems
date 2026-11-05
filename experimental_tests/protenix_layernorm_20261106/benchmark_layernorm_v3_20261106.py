"""Correct axes, masks and long-row parameter reductions on actual LN operands."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch
from protenix.model.layer_norm.layer_norm import fast_layer_norm_cuda_v2 as backend
from analyze_ln_abi_20261106 import decode, formula, formula_check, literal, sha, streaming_reference
from operator_replay_helpers_20261031AB import metrics, move_storage

HERE=Path(__file__).resolve().parent
L=Path('/root/xishiqing_protenix_precision_20261023')


def module(name):
    path=HERE/('layernorm_'+name+'.py')
    spec=importlib.util.spec_from_file_location('flag_gems.ops.ln_'+name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,required=True);args,_=parser.parse_known_args()
    torch.set_num_threads(8)
    modules={label:module(label) for label in ('axes_fixed','candidate_v2','candidate_v3')}
    state={'status':'RUNNING','source_sha256':sha(__file__),'operator_sources':{label:sha(HERE/('layernorm_'+label+'.py')) for label in modules},'formula_checks':formula_check(),'generic_checks':[],'cases':[],'new_H100_execution':False}
    target=args.output_root/'result.json'
    def save():target.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    save()
    generator=torch.Generator().manual_seed(20261106)
    cases=[((3,5),(5,),False),((2,3,7),(7,),False),((2,3,5),(3,5),False),
           ((5,),(5,),False),((17,33),(33,),True),((4097,7),(7,),False),
           ((31,129),(129,),False),((16387,65),(65,),False)]
    for shape,norm,strided in cases:
        size=list(shape);size[-1]*=2 if strided else 1
        x=torch.randn(size,generator=generator)[...,::2] if strided else torch.randn(size,generator=generator)
        dy=torch.randn(size,generator=generator)[...,::2] if strided else torch.randn(size,generator=generator)
        n=math.prod(norm);flat=x.double().reshape(-1,n);mean=flat.mean(-1);rstd=((flat-mean[:,None]).square().mean(-1)+1e-5).rsqrt()
        stored_mean,stored_rstd=mean.float(),rstd.float()
        for use_w,use_b in [(False,False),(False,True),(True,False),(True,True)]:
            w=torch.randn(norm,generator=generator) if use_w else None
            b=torch.randn(norm,generator=generator) if use_b else None
            expected=formula(dy,x,norm,stored_mean,stored_rstd,w,b)
            expected=[v.reshape(shape if i==0 else norm) if v is not None else None for i,v in enumerate(expected)]
            values=move_storage((dy,x,stored_mean,stored_rstd,w,b),'cuda')
            for mask in [(True,use_w,use_b),(True,False,False),(False,use_w,use_b),(False,False,False)]:
                g,xx,mm,rr,ww,bb=values
                actual=modules['candidate_v3'].layer_norm_backward(g,xx,norm,mm,rr,ww,bb,mask)
                errors=[]
                for enabled,a,e in zip(mask,actual,expected):
                    if not enabled:assert a is None;errors.append(None);continue
                    error=metrics(a.cpu(),e);assert error['status']=='finite'
                    assert error['absolute_L2'] <= 2e-5*math.sqrt(error['reference_squared'])+2e-10,error
                    errors.append(error)
                repeat=modules['candidate_v3'].layer_norm_backward(g,xx,norm,mm,rr,ww,bb,mask)
                assert literal(actual,repeat)
                state['generic_checks'].append({'shape':shape,'normalized_shape':norm,'strided':strided,'weight':use_w,'bias':use_b,'mask':mask,'errors':errors,'repeat_byte_equal':True})
            del values,actual,repeat
        save();print('PASS_GENERIC_LN',shape,norm,strided,flush=True)
    # Empty outer rows retain zero affine gradients without launching a zero grid.
    x=torch.empty((0,5),device='cuda');w=torch.ones(5,device='cuda');b=torch.zeros_like(w);stats=torch.empty(0,device='cuda')
    empty=modules['candidate_v3'].layer_norm_backward(x,x,(5,),stats,stats,w,b,(True,True,True))
    assert empty[0].shape==x.shape and all(bool((v==0).all()) for v in empty[1:])
    state['empty_outer_rows']='PASS'
    for precision in ('fp32','tf32'):
        capture=L/('ln_abi_capture_v2_'+precision+'_20261106')
        assert (capture/'exit_code').read_text()=='0'
        report=json.loads((capture/'msa0_replay.json').read_text());assert report['status']=='COMPLETE_LITERAL_MSA0_CHILD_REPLAY'
        torch.backends.cuda.matmul.allow_tf32=precision=='tf32'
        for entry in report['layer_norm_ABI']['packets']:
            path=Path(entry['path']);assert sha(path)==entry['sha256']
            packet=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            dy,x,norm,mean,rstd,w,b=decode(packet);mask=(True,w is not None,b is not None)
            moved=move_storage((dy,x,mean,rstd,w,b,packet['args']),'cuda')
            def call(label):
                if label=='model_CUDA':return getattr(backend,packet['backend'])(*moved[-1])
                gg,xx,mm,rr,ww,bb=moved[:6]
                return modules[label].layer_norm_backward(gg,xx,norm,mm,rr,ww,bb,mask)
            names=['model_CUDA','axes_fixed','candidate_v2','candidate_v3'];outputs={}
            for name in names:
                value=call(name);assert literal(value,call(name))
                outputs[name]=[v.cpu() if v is not None else None for v in value]
                if name=='model_CUDA':assert literal(outputs[name],packet['outputs'])
                for _ in range(2):call(name)
            times={name:[] for name in names}
            for iteration in range(7):
                order=names[iteration%len(names):]+names[:iteration%len(names)]
                for name in order:
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record();value=call(name);end.record();end.synchronize();times[name].append(begin.elapsed_time(end))
            assert literal(moved[:6],(dy,x,mean,rstd,w,b))
            del moved,value
            row={**entry,'source_precision':precision,'implementations':{}}
            for name in names:
                changed=dict(packet,outputs=outputs[name])
                errors,statistics=streaming_reference(changed)
                row['implementations'][name]={'errors':errors,'statistics':statistics,'device_event_ms':times[name],'median_device_event_ms':sorted(times[name])[3],'repeat_byte_equal':True}
            state['cases'].append(row);save();print('COMPLETE_REAL_LN_BENCH',precision,entry['input_shape'],entry['dy_nonzero'],flush=True)
    state['status']='COMPLETE_LAYERNORM_CANDIDATE_VALIDATION';save();print(state['status'],len(state['generic_checks']),len(state['cases']),flush=True)


if __name__=='__main__':main()
