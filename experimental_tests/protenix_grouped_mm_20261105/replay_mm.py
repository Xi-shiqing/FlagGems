"""Replay actual dW packets through three existing FlagGems MM implementations."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import faulthandler
faulthandler.enable()
HERE = Path(__file__).resolve().parent
PREVIOUS = Path('/mnt/cpfs/xishiqing/ppu_full_acceptance_20261005/precision_residual_20261019')
PACKETS = Path('/root/xishiqing_protenix_precision_20261023')

def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_module(label, directory):
    path = HERE/'grouped_work_mm.py' if label == 'grouped_work' else HERE/'grouped64_mm.py' if label == 'grouped64' else HERE/'grouped_mm.py' if label == 'grouped' else PREVIOUS / directory / 'flag_gems/ops/mm.py'
    name = 'flag_gems.ops.mm_comparison_' + label
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, dict(path=str(path), sha256=digest(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'dsw-766246-9578bd86-vmm4s'
    sys.path.insert(0, str(PREVIOUS.parent))
    from bmm_card_gate_20261015ag import acquire
    lock, reason, card = acquire('long_MM_actual_packets_20261105', HERE/'admission.json', 'exclusive', preferred=args.gpu)
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(card['gpu'])
        import torch
        torch.set_num_threads(4)
        import triton
        assert torch.__version__ == '2.9.0' and triton.__version__ == '3.6.0'
        assert triton.__file__.startswith('/usr/local/lib/python3.12/site-packages/')
        print('RUNTIME', torch.__version__, triton.__version__, triton.__file__, flush=True)
        sys.path.insert(0, str(PREVIOUS))
        from operator_replay_helpers_20261031AB import move_storage, metrics
        def literal(first, second):
            if torch.is_tensor(first):
                return first.shape == second.shape and first.dtype == second.dtype and first.detach().cpu().numpy().tobytes(order='C') == second.detach().cpu().numpy().tobytes(order='C')
            return len(first) == len(second) and all(literal(a, b) for a, b in zip(first, second))
        witness = torch.tensor([0., -0., 1.])
        assert literal(witness, witness.clone())
        assert not literal(witness, torch.tensor([-0., -0., 1.]))
        scalar_stride=torch.empty_strided((1,1),(2,2)).fill_(1)
        assert literal(scalar_stride, torch.ones(1,1))
        modules, identities = {}, {}
        for label, directory in [('original','gems_canonical_softmax_v2'),
                                 ('occupancy','gems_long_reduction_splitk_v2'),
                                 ('bounded_tree','gems_bounded_reduction_v1'), ('grouped', None), ('grouped64', None), ('grouped_work', None)]:
            modules[label], identities[label] = load_module(label, directory)
        print('MM_MODULES_IMPORTED', identities, flush=True)
        assert identities['bounded_tree']['sha256'] == '300371952fb3e670e38ec130d1ec82ceb4ed527c03009a131457eedbe78f4e7e'
        assert identities['original']['sha256'] == '5e1cf91e8b62929dd2d96990d7b2232f539fdd7a0e6c0d5ed79a0a45d551e7b8'
        assert identities['occupancy']['sha256'] == '6c3c087f6760a921759e0721144e1a2312c4b04de8e44797ccbf6bb6618fa1b9'
        state = dict(status='RUNNING', card=card, runtime_sources=identities,
                     source_sha256=digest(Path(__file__)), cases=[], generic_checks=[], new_H100_GPU_execution=False,
                     runtime={'torch': torch.__version__, 'torch_file': torch.__file__, 'triton': triton.__version__, 'triton_file': triton.__file__},
                     timing_scope='Complete FlagGems MM API, own allocations, five interleaved CUDA-event measurements after warmup')
        output_dir=HERE/'outputs';output_dir.mkdir(exist_ok=True)
        grouped = modules['grouped_work']
        selected_calls=[]
        original_route=grouped._long_reduction_splitk_mm
        def observed(*values):
            result=original_route(*values)
            if result is not None:selected_calls.append(values[3:])
            return result
        grouped._long_reduction_splitk_mm=observed
        generator=torch.Generator().manual_seed(20261105)
        generic_shapes=[(1,1,8209,False),(17,65,8192,False),(17,65,8193,False),
                        (33,17,8209,True),(65,33,16387,True),(64,257,10007,False),
                        (1024,128,4096,False),(512,512,8209,False),(128,512,65539,False),(256,1024,65539,True)]
        for M,N,K,strided in generic_shapes:
            width=2 if strided else 1
            ah=torch.randn((M,K*width),generator=generator)[:,::width]
            bh=torch.randn((K,N*width),generator=generator)[:,::width]
            reference=ah.double()@bh.double()
            def quantize(v):
                bits=v.contiguous().view(torch.int32).to(torch.int64)
                return ((bits+4095+((bits>>13)&1))&0xFFFFE000).to(torch.int32).view(torch.float32)
            quantized=quantize(ah).double()@quantize(bh).double()
            a,b=move_storage((ah,bh),'cuda')
            for mode in ('fp32','tf32'):
                torch.backends.cuda.matmul.allow_tf32=mode=='tf32'
                before=len(selected_calls);value=grouped.mm(a,b).cpu();selected=len(selected_calls)>before
                assert literal(value,grouped.mm(a,b).cpu())
                oracle=quantized if mode=='tf32' and N!=1 else reference
                error=metrics(value,oracle)
                assert error['relative_L2_percent'] < (0.02 if mode=='tf32' else 0.002),error
                if not selected:assert literal(value,modules['original'].mm(a,b).cpu())
                storage=torch.full((M,2*N),123.5,device='cuda');out=storage[:,::2]
                returned=grouped.mm_out(a,b,out=out)
                assert returned is out and literal(out.cpu(),value)
                assert bool((storage[:,1::2]==123.5).all())
                state['generic_checks'].append(dict(M=M,N=N,K=K,strided=strided,mode=mode,
                    selected=selected,error_vs_declared_reference=error,repeat_and_strided_mm_out='PASS'))
                (HERE/'result.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
            print('PASS_GENERIC_GROUPED_MM',M,N,K,strided,flush=True)
            del a,b,ah,bh,reference,quantized,storage,out,value
        plan=json.loads((HERE/'input_plan.json').read_text())
        for entry in plan['packets']:
            packet_path=Path(entry['source'])
            assert packet_path.stat().st_size==entry['bytes'] and digest(packet_path)==entry['sha256']
            print('LOADING_ACTUAL_PACKET', entry['sha256'], flush=True)
            payload=torch.load(packet_path,map_location='cpu',weights_only=False,mmap=True)
            assert payload['operator']=='aten.mm.default' and not payload['kwargs']
            ah,bh=payload['args']; assert ah.shape[1]>8192
            print('CPU64_REFERENCE', list(ah.shape), list(bh.shape), flush=True)
            reference=ah.double() @ bh.double()
            def rne(value):
                bits=value.contiguous().view(torch.int32).to(torch.int64)
                return ((bits+4095+((bits>>13)&1))&0xFFFFE000).to(torch.int32).view(torch.float32)
            quantized=rne(ah).double() @ rne(bh).double()
            a,b=move_storage((ah,bh),'cuda')
            original_inputs=(a.cpu().clone(),b.cpu().clone())
            source_mode=payload['source_precision'];step=payload['step'];call=payload['call']
            for mode in ('fp32','tf32'):
                torch.backends.cuda.matmul.allow_tf32=mode=='tf32'
                row=dict(input_precision=source_mode, execution_precision=mode,step=step,call=call,
                         shapes=[list(ah.shape),list(bh.shape)],strides=[list(ah.stride()),list(bh.stride())],
                         input_sha256=entry['sha256'],quantization_only_vs_CPU64=metrics(quantized,reference),implementations={})
                outputs={}
                for label,module in modules.items():
                    print('CALL_MM', source_mode, mode, step, label, flush=True)
                    value=module.mm(a,b).cpu()
                    assert literal(value,module.mm(a,b).cpu()) and torch.isfinite(value).all()
                    if label=='original' and source_mode==mode:
                        assert literal(value,payload['actual_outputs']), 'Saved default output was not reproduced'
                    parts=module._long_reduction_splitk_parts(a,b,None,a.shape[0],b.shape[1],a.shape[1]) if label!='original' else None
                    target=output_dir/f'{source_mode}_{mode}_step{step}_{label}.pt'
                    assert not target.exists();torch.save(value,target)
                    outputs[label]=value
                    row['implementations'][label]=dict(vs_CPU64=metrics(value,reference),
                        vs_RNE_operand_CPU64=metrics(value,quantized),parts=parts,
                        repeats_byte_equal=True,output_file=str(target),output_sha256=digest(target),device_ms=[])
                    for _ in range(2): module.mm(a,b)
                torch.cuda.synchronize()
                labels=list(modules)
                for iteration in range(5):
                    order=labels[iteration%len(labels):]+labels[:iteration%len(labels)]
                    for label in order:
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();result=modules[label].mm(a,b);end.record();end.synchronize()
                        row['implementations'][label]['device_ms'].append(start.elapsed_time(end))
                for label in labels:
                    values=row['implementations'][label]['device_ms']
                    row['implementations'][label]['median_device_ms']=sorted(values)[2]
                assert literal((a.cpu(),b.cpu()),original_inputs)
                row['input_values_unchanged']=True
                state['cases'].append(row)
                (HERE/'result.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
                print('COMPLETE_MM',source_mode,mode,step,call,flush=True)
            del a,b,ah,bh,payload,reference,quantized,original_inputs
        state['status']='COMPLETE'
        (HERE/'result.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
        print('COMPLETE_REAL_MM_COMPARISON',len(state['cases']),flush=True)
    finally:
        if lock.exists() and lock.read_text()==reason: lock.unlink()

if __name__=='__main__':main()
