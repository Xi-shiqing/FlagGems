"""Matched BMM shapes, neighboring dimensions, layouts, out, backward and update."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics
import sys
import time
from controls import controls, switch
from audit_routes import metrics, rne, tensor_meta

CASES = (
    ('pair_qk', (16,245,24,245)), ('pair_pv', (16,245,245,24)),
    ('confidence_vector', (2529,1,384,50)),
    ('local_qk', (64,32,32,32)), ('local_pv', (256,32,128,32)),
    ('k23', (2,65,23,24)), ('k31', (2,65,31,24)),
    ('k32', (2,65,32,24)), ('k33', (2,65,33,24)),
    ('n33', (2,65,127,33)), ('column_vector', (2,63,129,1)),
    ('rank_one', (2,67,1,29)),
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', choices=('ppu','h100'), required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    extra = os.environ.get('PROTENIX_EXTRA_SITE_PACKAGES')
    if extra:
        sys.path.append(extra)
    import numpy as np
    import torch
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    from bench_tf32_training import _evaluate, _inputs, LR
    if args.server == 'ppu':
        import flag_gems
        entries = controls(args.baseline)
        context = lambda: flag_gems.use_gems(exclude=[])
        modes = ('fp32','tf32','candidate_tf32')
    else:
        entries, context, modes = [], nullcontext, ('fp32','tf32')

    def configure(mode):
        switch(entries, mode == 'candidate_tf32')
        torch.backends.cuda.matmul.allow_tf32 = mode != 'fp32'
        torch.backends.cudnn.allow_tf32 = mode != 'fp32'

    results = []
    for name,(batch,M,K,N) in CASES:
        arrays = _inputs('bmm',(batch,M,K,N),101)
        aa,bb,dy = [torch.from_numpy(arrays[k]).double() for k in ('a','weight','upstream')]
        gold = {'output':aa@bb, 'grad_activation':dy@bb.transpose(-1,-2),
                'grad_weight':aa.transpose(-1,-2)@dy}
        updated = (bb-LR*gold['grad_weight']).float()
        gold.update({'updated_weight':updated, 'weight_update':updated-bb.float()})
        rounded = rne(aa.float()).double()@rne(bb.float()).double()
        row = {'name':name, 'shape':[batch,M,K,N],
               'inputs':{k:tensor_meta(torch.from_numpy(v)) for k,v in arrays.items()},
               'gold':{k:tensor_meta(v) for k,v in gold.items()}, 'modes':{}}
        with context():
            for mode in modes:
                configure(mode)
                value = _evaluate('bmm',arrays,'cuda',args.server,mode!='fp32')
                row['modes'][mode] = {'errors_vs_cpu_fp64':{
                    k:metrics(v,gold[k])|{'sha256':tensor_meta(v)['sha256']} for k,v in value.items()},
                    'forward_vs_rne_fp64':metrics(value['output'],rounded)}
            a = torch.from_numpy(arrays['a']).cuda()
            b = torch.from_numpy(arrays['weight']).cuda()
            # Non-contiguous, offset storage; poison output first and inspect exact values.
            ax = torch.empty((batch,M,K*2),device='cuda'); ax[...,::2].copy_(a); ax=ax[...,::2]
            bx = torch.empty((batch,K,N*2),device='cuda'); bx[...,::2].copy_(b); bx=bx[...,::2]
            out_base = torch.full((batch,M,N*2),float('nan'),device='cuda')
            out = out_base[...,::2]
            layout = {}
            for mode in modes:
                configure(mode)
                value = torch.bmm(ax,bx,out=out)
                assert value.data_ptr()==out.data_ptr() and bool(torch.isfinite(out).all())
                layout[mode] = tensor_meta(out)
                if args.server=='ppu' and mode=='candidate_tf32':
                    policy_gold = rounded if M>1 and N>1 and K>=32 else gold['output']
                    assert torch.allclose(out.cpu().double(),policy_gold,rtol=3e-4,atol=2e-4),name
            row['strided_out'] = layout
            if args.server=='ppu':
                configure('fp32'); switch(entries,True)
                shadow = _evaluate('bmm',arrays,'cuda',args.server,False)
                assert all(tensor_meta(v)['sha256']==row['modes']['fp32']['errors_vs_cpu_fp64'][k]['sha256']
                           for k,v in shadow.items()), name
                row['candidate_fp32_bitwise_equal_baseline']=True
            samples={mode:[] for mode in modes}
            for repetition in range(13):
                for mode in modes:
                    configure(mode)
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    torch.bmm(a,b)
                    torch.cuda.synchronize()
                    if repetition>=3:
                        samples[mode].append((time.perf_counter()-start)*1000)
            for mode in modes:
                row['modes'][mode].update({'median_inference_ms':statistics.median(samples[mode]),
                    'samples_ms':samples[mode], 'warmup':3,'repeats':10})
        results.append(row)
        args.output.write_text(json.dumps({'server':args.server,'seed':101,'cases':results,
            'status':'PASS' if len(results)==len(CASES) else 'PARTIAL',
            'protocol':'PCG64 matched operands, full FP64 forward/backward gold, FP32 rounded SGD update; timing forward only. Generic policy experiment, not exact native kernel emulation.'},indent=2)+'\n')
        print('BMM_POLICY',name,{m:round(row['modes'][m]['median_inference_ms'],5) for m in modes},flush=True)
    switch(entries,True)


if __name__=='__main__':
    main()
