"""Resolve actual full-gradient and SGD differences at the real cache boundary."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

os.environ.pop('PROTENIX_TF32_CAPTURE_DIR', None)
os.environ['PROTENIX_TF32_RESTORE_MODULES'] = ''

import torch
from bench_query_once import inputs, sha, metrics, LR, KEYS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', choices=('ppu', 'h100'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    s = 384
    arrays = inputs(s, 101 + s)
    params = [torch.from_numpy(arrays[key]).cuda().expand(1, s, s, 4, 32).contiguous().requires_grad_() if key != 'bias2'
              else torch.from_numpy(arrays[key]).cuda().requires_grad_() for key in KEYS]
    upstream = torch.from_numpy(arrays['upstream']).cuda().expand_as(params[0]).contiguous()
    bias1 = torch.zeros((1, s, 1, 1, s), device='cuda')
    prefix = 'cuda_' if args.server == 'h100' else ''
    reference = None
    result = {'server': args.server, 'shape': [1, s, s, 4, 32], 'seed': 101+s,
              'input_sha256': {key: sha(value.detach().cpu().numpy()) for key, value in zip((*KEYS, 'upstream'), (*params, upstream))},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'modes': {}}
    for name in ('fixed_rne_once', 'bounded_prequant', 'prequant_cached_ds'):
        module = importlib.import_module(prefix + name)
        output = module.triangle_attention(*params[:3], bias1, params[3], precision='tf32', layout='BNSHD')
        grads = torch.autograd.grad(output, params, upstream)
        current = {'output': output.detach().cpu()}
        for key, param, grad in zip(KEYS, params, grads):
            updated = param.detach().clone().add_(grad, alpha=-LR)
            current[key] = grad.detach().cpu()
            current[key + '_updated'] = updated.cpu()
            current[key + '_increment'] = (updated - param.detach()).cpu()
        if reference is None:
            reference = current
        result['modes'][name] = {key: {'vs_parent_rne': metrics(value, reference[key]), 'sha256': sha(value.numpy())} for key, value in current.items()}
        assert result['modes'][name]['output']['vs_parent_rne']['max_abs'] == 0.
        assert all(value['vs_parent_rne']['relative_l2'] < 1e-4 and value['vs_parent_rne']['finite'] for value in result['modes'][name].values())
        print('LARGEST_UPDATE_DIFF', args.server, name, max(value['vs_parent_rne']['relative_l2'] for value in result['modes'][name].values()), flush=True)
    result['status'] = 'PASS'
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
