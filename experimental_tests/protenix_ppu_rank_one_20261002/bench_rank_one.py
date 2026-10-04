"""Matched operands: real Torch entries, forward/backward/SGD, four precisions."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from controls import controls, switch


CASES = (
    ('mm_charge', 'mm', (2529, 1, 128)),
    ('mm_distance_inverse', 'mm', (327680, 1, 16)),
    ('addmm_bias', 'addmm', (257, 1, 128)),
    ('bmm_attention_rank1', 'bmm', (16, 245, 1, 24)),
    ('bmm_odd_sizes', 'bmm', (97, 7, 1, 33)),
    ('mm_k3_regression', 'mm', (129, 3, 16)),
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', choices=('ppu', 'h100'), required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    extra = os.environ.get('PROTENIX_EXTRA_SITE_PACKAGES')
    if extra:
        sys.path.append(extra)
    import torch
    from bench_tf32_training import _inputs, _evaluate, _sha, _train_step, LR
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    entries = []
    if args.server == 'ppu':
        import flag_gems
        entries = controls(args.baseline)
        context = lambda: flag_gems.use_gems(exclude=[])
        modes = ('fp32', 'tf32', 'candidate_tf32')
    else:
        context = nullcontext
        modes = ('fp32', 'tf32')

    def configure(mode):
        switch(entries, mode == 'candidate_tf32')
        torch.backends.cuda.matmul.allow_tf32 = mode != 'fp32'
        torch.backends.cudnn.allow_tf32 = mode != 'fp32'

    def error(value, reference):
        x, y = value.detach().cpu().double(), reference.detach().cpu().double()
        delta = x-y
        return {'relative_l2': float(delta.norm()/y.norm().clamp_min(1e-30)),
                'max_abs': float(delta.abs().max()), 'finite': bool(torch.isfinite(x).all()),
                'sha256': _sha(value.detach().cpu().numpy())}

    results = []
    for name, kind, shape in CASES:
        arrays = _inputs(kind, shape, 101)
        aa, ww, dy = [torch.from_numpy(arrays[k]).double() for k in ('a', 'weight', 'upstream')]
        bias_cpu = torch.from_numpy(arrays['bias']).double() if kind == 'addmm' else None
        if kind == 'bmm':
            output = aa @ ww
            ga, gw = dy @ ww.transpose(-1, -2), aa.transpose(-1, -2) @ dy
        else:
            output = aa @ ww.T
            ga, gw = dy @ ww, dy.T @ aa
            if kind == 'addmm':
                output += bias_cpu
        updated = (ww-LR*gw).float()
        gold = {'output': output, 'grad_activation': ga, 'grad_weight': gw,
                'updated_weight': updated, 'weight_update': updated-ww.float()}
        if bias_cpu is not None:
            gb = dy.sum(0)
            updated_bias = (bias_cpu-LR*gb).float()
            gold.update({'grad_bias': gb, 'updated_bias': updated_bias,
                         'bias_update': updated_bias-bias_cpu.float()})
        row = {'name': name, 'kind': kind, 'shape': list(shape),
               'input_sha256': {k:_sha(v) for k,v in arrays.items()},
               'gold_sha256': {k:_sha(v.numpy()) for k,v in gold.items()}, 'modes': {}}
        with context():
            for mode in modes:
                configure(mode)
                value = _evaluate(kind, arrays, 'cuda', args.server, mode != 'fp32')
                row['modes'][mode] = {'errors_vs_cpu_fp64': {k:error(v,gold[k]) for k,v in value.items()}}
            if args.server == 'ppu':
                switch(entries, True)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                shadow = _evaluate(kind, arrays, 'cuda', args.server, False)
                assert all(_sha(v.numpy()) == row['modes']['fp32']['errors_vs_cpu_fp64'][k]['sha256']
                           for k,v in shadow.items()), name
                row['candidate_fp32_bitwise_equal_baseline'] = True
                is_rank_one = shape[-2] == 1 if kind == 'bmm' else shape[1] == 1
                if is_rank_one:
                    assert row['modes']['candidate_tf32']['errors_vs_cpu_fp64']['output']['relative_l2'] < 1e-6
                else:
                    assert all(row['modes']['candidate_tf32']['errors_vs_cpu_fp64'][k]['sha256'] ==
                               row['modes']['tf32']['errors_vs_cpu_fp64'][k]['sha256'] for k in gold)
            activation = torch.from_numpy(arrays['a']).cuda().requires_grad_()
            weight0 = torch.from_numpy(arrays['weight']).cuda()
            weight = weight0.clone().requires_grad_()
            upstream = torch.from_numpy(arrays['upstream']).cuda()
            bias0 = torch.from_numpy(arrays['bias']).cuda() if kind == 'addmm' else None
            bias = bias0.clone().requires_grad_() if bias0 is not None else None
            inference = lambda: (torch.addmm(bias, activation, weight.T) if kind == 'addmm' else
                                 torch.bmm(activation, weight) if kind == 'bmm' else torch.mm(activation, weight.T))
            timings = {mode:{'inference_ms':[], 'training_ms':[]} for mode in modes}
            for repetition in range(13):
                for mode in modes:
                    configure(mode)
                    with torch.no_grad():
                        weight.copy_(weight0)
                        if bias is not None:
                            bias.copy_(bias0)
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        before = time.perf_counter()
                        inference()
                        torch.cuda.synchronize()
                        inference_ms = (time.perf_counter()-before)*1000
                    before = time.perf_counter()
                    _train_step(kind, activation, weight, upstream, bias)
                    torch.cuda.synchronize()
                    training_ms = (time.perf_counter()-before)*1000
                    if repetition >= 3:
                        timings[mode]['inference_ms'].append(inference_ms)
                        timings[mode]['training_ms'].append(training_ms)
            for mode, values in timings.items():
                row['modes'][mode].update({'samples_ms':values,
                    'median_inference_ms': statistics.median(values['inference_ms']),
                    'median_training_ms': statistics.median(values['training_ms']),
                    'warmup':3, 'repeats':10})
        results.append(row)
        args.output.write_text(json.dumps({'server':args.server,'seed':101,'cases':results,
            'status':'PASS' if len(results)==len(CASES) else 'PARTIAL',
            'protocol':'Matched PCG64 integer operands; float32 storage; seed101. CPU FP64 full-tensor forward/gradient gold. Updated parameters reference one-rounding FP32 SGD. Three warmups and ten alternating measured repetitions; reset parameters outside timing.'},indent=2)+'\n')
        print('RANK_ONE_BENCH',name,{m:row['modes'][m]['median_inference_ms'] for m in modes},flush=True)
    switch(entries,True)


if __name__ == '__main__':
    main()
