#!/usr/bin/env python3
"""Pair baseline/candidate public MM and BMM on audited real activations."""
import argparse
import ast
import importlib
import json
from pathlib import Path
import statistics
import time

import torch
import flag_gems
from bench_vendor_mm import load_helper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--capture', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--numeric-candidate', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assert torch.utils.deterministic.fill_uninitialized_memory
    helper = load_helper(args.helper)
    modules, originals, candidates = {}, {}, {}
    for name in ('mm', 'bmm'):
        module = importlib.import_module('flag_gems.ops.' + name)
        modules[name] = module
        names = (name, '_bmm_rne_once', '_direct_mm') if args.numeric_candidate else (name, '_bmm_rne_once')
        candidates[name] = {n: getattr(module, n) for n in names if hasattr(module, n)}
        source = args.baseline / ('flag_gems/ops/' + name + '.py')
        namespace = dict(vars(module))
        nodes = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in candidates[name]]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        originals[name] = {n: namespace[n] for n in candidates[name]}

    def choose(name, variant):
        functions = originals[name] if variant == 'baseline' else candidates[name]
        for key, function in functions.items():
            setattr(modules[name], key, function)
        # Registered functions retain the original wrapper object. Use its
        # module-level RNE helper and call the selected public implementation.
        return getattr(modules[name], name)

    rows = []
    with torch.inference_mode(), flag_gems.use_gems(exclude=[]):
        for capture_root in args.capture:
            capture = json.loads((capture_root / 'activation_capture.json').read_text())
            for key, record in capture['ops']['selected'].items():
                name = key.split('_')[0]
                if name not in modules:
                    continue
                a, b = helper._load_inputs(capture_root, record, 'cuda')
                for precision in ('fp32', 'tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = precision == 'tf32'
                    outputs = {}
                    for variant in ('baseline', 'candidate'):
                        call = choose(name, variant)
                        for _ in range(5):
                            output = call(a, b)
                        torch.cuda.synchronize()
                        outputs[variant] = output.clone()
                    equal = torch.equal(outputs['baseline'], outputs['candidate'])
                    errors = helper._metrics(outputs['candidate'], outputs['baseline'])
                    if precision == 'fp32' or not args.numeric_candidate:
                        assert equal, (capture_root, key, precision)
                    else:
                        assert errors['finite'] and errors['relative_l2'] < 1e-5
                    samples = {v: [] for v in ('baseline', 'candidate')}
                    loops = 5 if outputs['candidate'].numel() < 262144 else 2
                    for repeat in range(15):
                        order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
                        for variant in order:
                            call = choose(name, variant)
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            for _ in range(loops):
                                output = call(a, b)
                            torch.cuda.synchronize()
                            samples[variant].append((time.perf_counter() - start) * 1000 / loops)
                    row = {'capture': capture_root.name, 'op': key, 'precision': precision,
                           'inputs': [helper._metadata(t) for t in (a, b)],
                           'output': helper._metadata(outputs['candidate']), 'bitwise_equal': equal,
                           'candidate_vs_baseline': errors,
                           'samples_wall_ms': samples,
                           'medians_wall_ms': {v: statistics.median(s) for v, s in samples.items()}}
                    rows.append(row)
                    print(json.dumps({k: row[k] for k in ('capture', 'op', 'precision', 'medians_wall_ms')}), flush=True)
    args.out.write_text(json.dumps({'status': 'PASS', 'deterministic_algorithms': True,
        'fill_uninitialized_memory': True, 'candidate_source': flag_gems.__file__,
        'numerical_candidate': args.numeric_candidate,
        'scope': 'FlagGems implementation entrypoints; allocations/fill/preparation included; outer Torch dispatch excluded', 'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
