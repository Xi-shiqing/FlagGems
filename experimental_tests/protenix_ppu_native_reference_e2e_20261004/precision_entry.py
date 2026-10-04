"""Frozen native modes plus explicit TriAttention-only counterfactuals."""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import runpy
import statistics
import sys

from runtime_precision import audit_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boundary-harness', type=Path, required=True)
    parser.add_argument('--server', choices=('ppu', 'h100'), required=True)
    parser.add_argument('--reference-pair', type=Path, required=True)
    parser.add_argument('--paired-output', type=Path, required=True)
    parser.add_argument('--per-mode-warmup', type=int, default=1)
    parser.add_argument('--per-mode-repeats', type=int, default=3)
    parser.add_argument('--variants', nargs='+', choices=('fp32', 'tf32', 'tf32_triangle_ieee', 'tf32_triangle_corrected'))
    args, remaining = parser.parse_known_args()
    import torch
    from runner import batch_inference
    variants = ('fp32', 'tf32') if args.server == 'ppu' else ('fp32', 'tf32', 'tf32_triangle_ieee', 'tf32_triangle_corrected')
    if args.variants:
        assert all(v in variants for v in args.variants)
        variants = tuple(args.variants)
    order, audit, calls, checkpoints = [], {}, Counter(), {}
    active, first = [None], [False]
    native = None
    if args.server == 'h100':
        import cuequivariance_ops_torch as cue
        from triangle_adapter import triangle_attention as corrected
        native = cue.triangle_attention

        def triangle(*values, **kw):
            mode = active[0]
            if first[0]:
                calls[mode] += 1
            if mode == 'tf32_triangle_corrected':
                return corrected(*values, **kw)
            if mode == 'tf32_triangle_ieee':
                previous = torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    return native(*values, **kw)
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = previous
            return native(*values, **kw)
        cue.triangle_attention = triangle
    original = batch_inference.get_default_runner
    original_load = torch.load
    model = remaining[remaining.index('--model') + 1]

    def load(path, *a, **kw):
        value = original_load(path, *a, **kw)
        if isinstance(path, (str, Path)) and Path(path).stem == model:
            digest = hashlib.sha256()
            with Path(path).open('rb') as handle:
                for chunk in iter(lambda: handle.read(8*1024**2), b''):
                    digest.update(chunk)
            checkpoints[str(path)] = digest.hexdigest()
        return value
    torch.load = load

    def initialize(*a, **kw):
        runner = original(*a, **kw)
        predict = runner.predict
        def run(*a, **kw):
            mode = variants[len(order) % len(variants)]
            order.append(mode)
            active[0], first[0] = mode, mode not in audit
            torch.backends.cuda.matmul.allow_tf32 = mode != 'fp32'
            torch.backends.cudnn.allow_tf32 = mode != 'fp32'
            if args.server == 'ppu':
                os.environ['PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION'] = 'ieee' if mode == 'fp32' else 'tf32'
            try:
                with audit_prediction(audit, mode) if first[0] else nullcontext():
                    return predict(*a, **kw)
            finally:
                first[0] = False
        runner.predict = run
        return runner
    batch_inference.get_default_runner = initialize
    sys.argv = [str(args.boundary_harness), *remaining]
    runpy.run_path(str(args.boundary_harness), run_name='__main__')
    bench = json.loads((args.paired_output.parent / 'benchmark.json').read_text())
    refs = json.loads(args.reference_pair.read_text())['summary']
    assert len(order) == len(bench['records']) and checkpoints
    summary = {}
    for mode in variants:
        rows = [r for r, m in zip(bench['records'], order) if m == mode and r['phase'] == 'repeat']
        warm = [r for r, m in zip(bench['records'], order) if m == mode and r['phase'] == 'warmup']
        assert len(rows) == args.per_mode_repeats and len(warm) == args.per_mode_warmup
        assert all(r['prediction_digest'] == rows[0]['prediction_digest'] and r['five_confidence_metrics'] == rows[0]['five_confidence_metrics'] for r in rows + warm), mode
        if mode in ('fp32', 'tf32'):
            assert rows[0]['prediction_digest'] == refs[mode]['prediction_digest'], mode
            assert rows[0]['five_confidence_metrics'] == refs[mode]['five_confidence_metrics'], mode
        if args.server == 'h100':
            assert calls[mode] > 0, mode
        summary[mode] = {'count': len(rows), 'warmup_count': len(warm),
            'median_seconds': statistics.median(r['wall_seconds'] for r in rows),
            'samples_seconds': [r['wall_seconds'] for r in rows],
            'prediction_digest': rows[0]['prediction_digest'],
            'five_confidence_metrics': rows[0]['five_confidence_metrics']}
    args.paired_output.write_text(json.dumps({'status': 'PASS', 'order': order, 'summary': summary,
        'triangle_calls_first_warmup': dict(calls), 'checkpoint_sha256': checkpoints,
        'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'],
        'scope': 'Frozen FP32/TF32 predictions exactly retained. IEEE triangle is mixed precision; corrected triangle is custom CUDA RNE-once, not native CUE and not a query-only binary fix. Inference screen only.'}, indent=2) + '\n')
    (args.paired_output.parent / 'runtime_precision.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('NATIVE_REFERENCE_E2E_PASS', model, {m: v['median_seconds'] for m, v in summary.items()}, flush=True)


if __name__ == '__main__':
    main()
