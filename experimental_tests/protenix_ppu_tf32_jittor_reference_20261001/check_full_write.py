#!/usr/bin/env python3
"""Poison every new buffer to prove complete writes, including odd edge tiles."""
import argparse
import ast
import importlib
import json
from pathlib import Path

import numpy as np
import torch
import flag_gems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    assert torch.utils.deterministic.fill_uninitialized_memory
    controls, modules, counts = [], {}, [0]
    for name in ('mm', 'bmm'):
        module = importlib.import_module('flag_gems.ops.' + name)
        modules[name] = module
        source = args.baseline / ('flag_gems/ops/' + name + '.py')
        nodes = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in (name, '_bmm_rne_once')]
        namespace = dict(vars(module))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        for node in nodes:
            function = getattr(module, node.name)
            controls.append((function, namespace[node.name].__code__, function.__code__))
        original = module.empty_full_write
        def poisoned(*a, _original=original, **kw):
            tensor = _original(*a, **kw)
            tensor.fill_(float('nan'))
            counts[0] += 1
            return tensor
        module.empty_full_write = poisoned
    rng = np.random.default_rng(101)
    def tensor(shape):
        return torch.from_numpy(rng.integers(-256, 257, size=shape, dtype=np.int32).astype(np.float32) / 256).cuda()
    cases = []
    for m, k, n in [(1, 97, 173), (17, 97, 173), (65, 97, 173), (129, 65, 257)]:
        cases.append(('mm', tensor((m, k)), tensor((n, k)).T, False))
    for batch, m, k, n in [(3, 17, 37, 65), (5, 269, 373, 509), (197, 1, 384, 50)]:
        cases.append(('bmm', tensor((batch, m, k)), tensor((batch, n, k)).transpose(-1, -2), False))
    a, b = tensor((5, 269, 373)), tensor((1, 509, 373)).transpose(-1, -2).expand(5, -1, -1)
    cases.append(('bmm', a, b, True))
    rows = []
    with torch.inference_mode(), flag_gems.use_gems(exclude=[]):
        for name, a, b, with_out in cases:
            for enabled in (False, True):
                torch.backends.cuda.matmul.allow_tf32 = enabled
                outputs = {}
                for variant in ('baseline', 'candidate'):
                    for function, original, candidate in controls:
                        function.__code__ = original if variant == 'baseline' else candidate
                    if with_out:
                        batch, m, _ = a.shape
                        n = b.shape[-1]
                        guarded = torch.full((batch, m, n + 4), 99.0, device='cuda')
                        out = guarded[:, :, 2:-2]
                        result = torch.bmm(a, b, out=out)
                        assert result is out
                        assert torch.all(guarded[:, :, :2] == 99) and torch.all(guarded[:, :, -2:] == 99)
                    else:
                        result = getattr(torch, name)(a, b)
                    outputs[variant] = result.clone()
                assert torch.equal(outputs['baseline'], outputs['candidate']), (name, a.shape, b.shape, enabled)
                assert torch.isfinite(outputs['candidate']).all()
                rows.append({'op': name, 'shapes': [list(a.shape), list(b.shape)],
                             'tf32': enabled, 'out_guarded': with_out, 'bitwise_equal': True})
    assert counts[0] > 0
    args.out.write_text(json.dumps({'status': 'PASS', 'poisoned_buffers': counts[0],
        'deterministic_algorithms': True, 'fill_uninitialized_memory': True, 'rows': rows}, indent=2) + '\n')
    print('FULL_WRITE_COVERAGE_PASS', len(rows), 'cases;', counts[0], 'poisoned buffers')


if __name__ == '__main__':
    main()
