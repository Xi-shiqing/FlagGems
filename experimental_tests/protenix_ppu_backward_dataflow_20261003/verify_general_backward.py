"""Independent groups, masks, batch/head dimensions, and forced budget boundaries."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.pop('PROTENIX_TF32_CAPTURE_DIR', None)
os.environ['PROTENIX_TF32_RESTORE_MODULES'] = ''
os.environ['FLAG_GEMS_TRIANGLE_ATTENTION_FP16_TRAINING_SCRATCH'] = '0'

import numpy as np
import torch
from bench_query_once import metrics, sha, LR

CASES = (((2, 5, 67, 2, 16), 'default'), ((1, 9, 129, 4, 64), 'default'),
         ((1, 7, 145, 8, 32), 'default'), ((2, 7, 67, 2, 16), 'three_groups'),
         ((1, 3, 67, 2, 16), 'smaller_than_one_group'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', choices=('ppu', 'h100'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    prefix = 'cuda_' if args.server == 'h100' else ''
    modules = {name: importlib.import_module(prefix + name) for name in ('fixed_rne_once', 'bounded_prequant', 'prequant_cached_ds')}
    result = {'status': 'RUNNING', 'server': args.server, 'cases': [],
              'scope': 'Independent CPU-PCG64 groups (not replicated), masked keys, B>1, H2/4/8, D16/32/64, layout permutation and forced private budget tests. Original CUE is not a training reference here; parent is the already tested custom RNE kernel.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for index, (shape, budget_test) in enumerate(CASES):
        b, n, s, h, d = shape
        rng = np.random.Generator(np.random.PCG64(401 + index))
        arrays = {key: rng.integers(-32768, 32768, size=shape, dtype=np.int32).astype(np.float32) * 2.0**-16 for key in ('q', 'k', 'v')}
        arrays['bias2'] = rng.integers(-32768, 32768, size=(b, 1, h, s, s), dtype=np.int32).astype(np.float32) * 2.0**-16
        arrays['upstream'] = rng.integers(-32768, 32768, size=shape, dtype=np.int32).astype(np.float32) * 2.0**-12
        valid = rng.integers(0, 3, size=(b, n, 1, 1, s), dtype=np.int32) != 0
        valid[..., 0] = True
        arrays['bias1'] = (~valid).astype(np.float32) * -1e9
        keys = ('q', 'k', 'v', 'bias2')
        gold_params = [torch.from_numpy(arrays[key]).double().requires_grad_() for key in keys]
        q, k, v, bias = gold_params
        q, k, v = [value.permute(0, 1, 3, 2, 4) for value in (q, k, v)]
        gold = ((q @ k.transpose(-1, -2)) + bias + torch.from_numpy(arrays['bias1']).double()).softmax(-1) @ v
        gold = gold.permute(0, 1, 3, 2, 4)
        gold_grads = torch.autograd.grad(gold, gold_params, torch.from_numpy(arrays['upstream']).double())
        row = {'shape': shape, 'seed': 401 + index, 'rng': 'NumPy PCG64', 'input_sha256': {key: sha(value) for key, value in arrays.items()},
               'budget_test': budget_test, 'modes': {}}
        reference = {}
        for precision in ('ieee', 'tf32'):
            for name, module in modules.items():
                budget_name = '_INPUT_CACHE_BUDGET_BYTES' if name == 'bounded_prequant' else '_DS_CACHE_BUDGET_BYTES'
                saved_budget = getattr(module, budget_name, None)
                if saved_budget is not None and budget_test != 'default':
                    bytes_per_group = 4*s*h*d*4 if name == 'bounded_prequant' else h*s*s*4
                    setattr(module, budget_name, bytes_per_group * 3 if budget_test == 'three_groups' else bytes_per_group - 1)
                params = [torch.from_numpy(arrays[key]).cuda().requires_grad_() for key in keys]
                bias1 = torch.from_numpy(arrays['bias1']).cuda()
                upstream = torch.from_numpy(arrays['upstream']).cuda()
                allocations = []
                empty = torch.empty
                def track(*a, **kw):
                    value = empty(*a, **kw)
                    if value.is_cuda:
                        allocations.append({'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': value.numel()*value.element_size()})
                    return value
                try:
                    # BNHSD strided views exercise the public normalization path.
                    views = [value.permute(0, 1, 3, 2, 4) for value in params[:3]]
                    with patch('torch.empty', track):
                        output = module.triangle_attention(*views, bias1, params[3], precision=precision, layout='BNHSD').permute(0, 1, 3, 2, 4)
                        grads = torch.autograd.grad(output, params, upstream)
                    values = {'output': output.detach().cpu(), **{key: grad.detach().cpu() for key, grad in zip(keys, grads)}}
                    if name == 'fixed_rne_once':
                        reference[precision] = values
                    original = reference[precision]
                    metrics_row = {'output_vs_fp64': metrics(output, gold), 'output_vs_parent': metrics(output, original['output']),
                                   'gradients': {}, 'allocations': allocations,
                                   'private_budget_bytes': getattr(module, budget_name, None)}
                    for key, param, grad, g in zip(keys, params, grads, gold_grads):
                        update = param.detach().clone().add_(grad, alpha=-LR) - param.detach()
                        parent_update = param.detach().cpu().clone().add_(original[key], alpha=-LR) - param.detach().cpu()
                        metrics_row['gradients'][key] = {'vs_fp64': metrics(grad, g), 'vs_parent': metrics(grad, original[key]),
                            'realized_update_vs_parent': metrics(update, parent_update), 'sha256': sha(grad.detach().cpu().numpy())}
                        assert metrics_row['gradients'][key]['vs_parent']['relative_l2'] < (1e-4 if precision == 'tf32' else 1e-12)
                        assert metrics_row['gradients'][key]['realized_update_vs_parent']['relative_l2'] < 1e-4
                        assert metrics_row['gradients'][key]['vs_fp64']['finite']
                    assert metrics_row['output_vs_parent']['max_abs'] == 0.
                    if name == 'bounded_prequant':
                        assert all(value['bytes'] <= module._INPUT_CACHE_BUDGET_BYTES for value in allocations)
                    elif name == 'prequant_cached_ds':
                        scratch = [value for value in allocations if len(value['shape']) == 5]
                        assert all(value['dtype'] == 'torch.float32' and value['bytes'] <= module._DS_CACHE_BUDGET_BYTES for value in scratch)
                    row['modes'][precision + ':' + name] = metrics_row
                finally:
                    if saved_budget is not None:
                        setattr(module, budget_name, saved_budget)
            print('GENERAL_BACKWARD_CASE', args.server, shape, budget_test, precision, flush=True)
        result['cases'].append(row)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    result['status'] = 'PASS'
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('GENERAL_BACKWARD_ALL_PASS', args.server, flush=True)


if __name__ == '__main__':
    main()
