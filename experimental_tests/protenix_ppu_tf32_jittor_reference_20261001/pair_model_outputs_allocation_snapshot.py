#!/usr/bin/env python3
"""Alternate allocation controls and strict FP32 in one full-FlagGems model."""
import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import runpy
import statistics
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--harness', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--paired-output', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    assert '--use-flag-gems' in remaining
    import torch
    import flag_gems
    from runner import batch_inference
    controls = []
    allocation_counts = {}
    active = ['initialization']
    for name in ('mm', 'bmm'):
        module = importlib.import_module('flag_gems.ops.' + name)
        assert getattr(flag_gems, name) is getattr(module, name)
        source = args.baseline / ('flag_gems/ops/' + name + '.py')
        nodes = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in (name, '_bmm_rne_once')]
        namespace = dict(vars(module))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        for node in nodes:
            function = getattr(module, node.name)
            assert not function.__closure__
            controls.append((function, namespace[node.name].__code__, function.__code__))
        allocate = module.empty_full_write
        def traced_allocate(*a, _allocate=allocate, **kw):
            allocation_counts[active[0]] = allocation_counts.get(active[0], 0) + 1
            return _allocate(*a, **kw)
        module.empty_full_write = traced_allocate
    variants = ('baseline_tf32', 'candidate_tf32', 'fp32')
    order = []
    original_init = batch_inference.get_default_runner

    def initialize(*a, **kw):
        runner = original_init(*a, **kw)
        original_predict = runner.predict
        def predict(*a, **kw):
            variant = variants[len(order) % len(variants)]
            order.append(variant)
            active[0] = variant
            for function, baseline_code, candidate_code in controls:
                function.__code__ = baseline_code if variant == 'baseline_tf32' else candidate_code
            torch.backends.cuda.matmul.allow_tf32 = variant != 'fp32'
            torch.backends.cudnn.allow_tf32 = variant != 'fp32'
            os.environ['PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION'] = 'ieee' if variant == 'fp32' else 'tf32'
            return original_predict(*a, **kw)
        runner.predict = predict
        return runner

    batch_inference.get_default_runner = initialize
    sys.argv = [str(args.harness), *remaining]
    try:
        runpy.run_path(str(args.harness), run_name='__main__')
    finally:
        for function, _, candidate_code in controls:
            function.__code__ = candidate_code
    bench = json.loads((args.paired_output.parent / 'benchmark.json').read_text())
    assert bench['environment']['use_flag_gems'] is True
    assert len(order) == len(bench['records'])
    summary = {}
    for variant in variants:
        records = [r for r, v in zip(bench['records'], order) if v == variant and r['phase'] == 'repeat']
        precision = 'fp32' if variant == 'fp32' else 'tf32'
        reference = json.loads((args.references / precision / 'benchmark.json').read_text())
        ref = next(r for r in reference['records'] if r['phase'] == 'repeat')
        assert all(r['prediction_digest'] == ref['prediction_digest'] for r in records), variant
        assert all(r['five_confidence_metrics'] == ref['five_confidence_metrics'] for r in records), variant
        values = [r['wall_seconds'] for r in records]
        summary[variant] = {'count': len(values), 'median_seconds': statistics.median(values),
                            'samples_seconds': values, 'prediction_and_confidence_equal_reference': True}
    assert allocation_counts.get('candidate_tf32', 0) > 0
    assert allocation_counts.get('baseline_tf32', 0) == allocation_counts.get('fp32', 0) == 0
    args.paired_output.write_text(json.dumps({'status': 'PASS', 'full_flag_gems': True,
        'order': order, 'summary': summary, 'full_write_allocation_counts': allocation_counts,
        'scope': 'One initialized model; alternating baseline TF32/candidate TF32/strict FP32'}, indent=2) + '\n')
    print('PAIRED_MODEL_PASS', summary, flush=True)


if __name__ == '__main__':
    main()
