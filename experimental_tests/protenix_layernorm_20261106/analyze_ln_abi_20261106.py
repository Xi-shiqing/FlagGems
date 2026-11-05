"""Replay actual LN ABI and stream all three CPU64 derivative comparisons."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from protenix.model.layer_norm.layer_norm import fast_layer_norm_cuda_v2 as backend
from operator_replay_helpers_20261031AB import fixed_input_reference, metrics, move_storage


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(data)
    return result.hexdigest()


def literal(a, b):
    if torch.is_tensor(a):
        return (a.shape == b.shape and a.dtype == b.dtype and
                a.detach().cpu().numpy().tobytes() == b.detach().cpu().numpy().tobytes())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(literal(x,y) for x,y in zip(a,b))
    return a == b


def decode(packet):
    args = packet['args']
    label = packet['backend']
    weight = args[5] if label in ('backward_with_weight_affine','backward_with_both_affine') else None
    bias = args[-2] if label in ('backward_with_bias_affine','backward_with_both_affine') else None
    return args[0], args[3], args[4], args[1], args[2], weight, bias


def formula(dy,x,shape,mean,rstd,weight,bias):
    # Reuse the shared saved-statistics formula. Flatten only normalized axes.
    columns = math.prod(shape)
    args = (dy.reshape(-1,columns),x.reshape(-1,columns),(columns,),
            mean.reshape(-1,1),rstd.reshape(-1,1),
            weight.reshape(-1) if weight is not None else None,
            bias.reshape(-1) if bias is not None else None,
            (True,weight is not None,bias is not None))
    return fixed_input_reference('aten.native_layer_norm_backward.default',args,{})


def formula_check():
    result = []
    for full,shape in [((3,7),(7,)),((2,3,5),(3,5)),((4,1),(1,))]:
        generator = torch.Generator().manual_seed(20261106)
        base = torch.randn(full,dtype=torch.float64,generator=generator)
        dy = torch.randn(full,dtype=torch.float64,generator=generator)
        for has_w,has_b in [(False,False),(False,True),(True,False),(True,True)]:
            x=base.clone().requires_grad_()
            w=torch.randn(shape,dtype=torch.float64,generator=generator,requires_grad=True) if has_w else None
            b=torch.randn(shape,dtype=torch.float64,generator=generator,requires_grad=True) if has_b else None
            y=torch.nn.functional.layer_norm(x,shape,w,b,1e-5)
            expected=torch.autograd.grad(y,[v for v in (x,w,b) if v is not None],dy)
            flat=x.detach().reshape(-1,math.prod(shape));mean=flat.mean(-1)
            rstd=((flat-mean[:,None]).square().mean(-1)+1e-5).rsqrt()
            actual=formula(dy,x.detach(),shape,mean,rstd,w,b)
            values=[]
            for a,e in zip([v for v in actual if v is not None],expected):
                error=metrics(a.reshape(e.shape),e)
                assert error['max_abs']<2e-12,error
                values.append(error)
            result.append({'shape':full,'normalized_shape':shape,'weight':has_w,'bias':has_b,'errors':values})
    return result


def streaming_reference(packet):
    dy,x,shape,mean,rstd,weight,bias=decode(packet)
    columns=math.prod(shape);rows=x.numel()//columns
    x=x.reshape(rows,columns);dy=dy.reshape(rows,columns)
    outputs=packet['outputs'];actual_dx=outputs[0].reshape(rows,columns)
    state={mode:{'dx':{'difference_squared':0.,'reference_squared':0.,'actual_squared':0.,'elements':0,'max_abs':0.},
                 'dw':torch.zeros(columns,dtype=torch.float64) if weight is not None else None,
                 'db':torch.zeros(columns,dtype=torch.float64) if bias is not None else None}
           for mode in ('saved_statistics','fresh_statistics')}
    stats={'mean':{'difference_squared':0.,'reference_squared':0.},'rstd':{'difference_squared':0.,'reference_squared':0.}}
    for start in range(0,rows,max(1,1048576//columns)):
        end=min(rows,start+max(1,1048576//columns))
        xx=x[start:end].double();gg=dy[start:end].double()
        fresh_mean=xx.mean(-1);fresh_rstd=((xx-fresh_mean[:,None]).square().mean(-1)+packet['args'][-1]).rsqrt()
        mm=mean[start:end].double();rr=rstd[start:end].double()
        for name,a,b in [('mean',mm,fresh_mean),('rstd',rr,fresh_rstd)]:
            value=metrics(a,b)
            for key in ('difference_squared','reference_squared'):stats[name][key]+=value[key]
        for mode,m,r in [('saved_statistics',mm,rr),('fresh_statistics',fresh_mean,fresh_rstd)]:
            dx,dw,db=formula(gg,xx,(columns,),m,r,weight,bias)
            value=metrics(actual_dx[start:end],dx)
            assert value['status']=='finite'
            for key in ('difference_squared','reference_squared','actual_squared','elements'):state[mode]['dx'][key]+=value[key]
            state[mode]['dx']['max_abs']=max(state[mode]['dx']['max_abs'],value['max_abs'])
            if dw is not None:state[mode]['dw']+=dw
            if db is not None:state[mode]['db']+=db
    result={}
    for mode,entry in state.items():
        value=entry['dx'];value['relative_L2_percent']=100*math.sqrt(value['difference_squared']/value['reference_squared']) if value['reference_squared'] else None
        result[mode]={'dx':value,
            'dw':metrics(outputs[1].reshape(-1),entry['dw']) if weight is not None else {'status':'not_requested'},
            'db':metrics(outputs[2].reshape(-1),entry['db']) if bias is not None else {'status':'not_requested'}}
    for value in stats.values():value['relative_L2_percent']=100*math.sqrt(value['difference_squared']/value['reference_squared']) if value['reference_squared'] else None
    return result,stats


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--precision',choices=('fp32','tf32'),required=True);parser.add_argument('--output-root',type=Path,required=True);args,_=parser.parse_known_args()
    torch.set_num_threads(8)
    assert torch.__version__=='2.10.0'
    capture=Path('/root/xishiqing_protenix_precision_20261023')/('ln_abi_capture_'+args.precision+'_20261106')
    assert (capture/'exit_code').read_text()=='0'
    report=json.loads((capture/'msa0_replay.json').read_text());assert report['status']=='COMPLETE_LITERAL_MSA0_CHILD_REPLAY'
    checks=formula_check()
    state={'status':'RUNNING','source_precision':args.precision,'arithmetic':'FP32 LayerNorm in both model precision modes','formula_checks':checks,'source_sha256':sha(__file__),'capture_sha256':sha(capture/'msa0_replay.json'),'cases':[],'new_H100_execution':False,'H100_same_operand_results':None}
    target=args.output_root/'result.json'
    def save():target.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    save()
    for entry in report['layer_norm_ABI']['packets']:
        path=Path(entry['path']);assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
        packet=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        values=move_storage(packet['args'],'cuda');original=[v.detach().cpu().clone() if torch.is_tensor(v) else v for v in values]
        operation=getattr(backend,packet['backend'])
        with torch.no_grad():
            first=operation(*values);assert literal(first,packet['outputs'])
            second=operation(*values);assert literal(first,second)
            assert literal(values,original)
            for _ in range(3):operation(*values)
            times=[]
            for _ in range(7):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();value=operation(*values);end.record();end.synchronize();times.append(begin.elapsed_time(end))
        assert literal(values,original)
        del values,original,first,second,value
        math_results,statistics=streaming_reference(packet)
        row={**entry,'ABI_reproduced_byte_equal':True,'repeat_byte_equal':True,'inputs_unchanged':True,
             'device_event_ms':times,'median_device_event_ms':sorted(times)[3],
             'reference_errors':math_results,'forward_statistics_error':statistics}
        state['cases'].append(row);save();print('COMPLETE_LN_ABI',args.precision,packet['backend'],entry['input_shape'],flush=True)
    state['status']='COMPLETE_ACTUAL_LN_ABI_REPLAY_AND_MATH';save();print(state['status'],len(state['cases']),flush=True)


if __name__=='__main__':main()
