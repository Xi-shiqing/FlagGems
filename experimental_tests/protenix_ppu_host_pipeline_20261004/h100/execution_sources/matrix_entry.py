"""Alternate original and allocation-only candidates in one initialized model."""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import importlib
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
    parser.add_argument('--per-mode-repeats', type=int, default=5)
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--poison', action='store_true')
    args, remaining = parser.parse_known_args()
    import torch
    from runner import batch_inference
    allocator = None
    if args.server == 'ppu':
        from flag_gems.utils import full_write_allocation as allocator
    variants = tuple(args.variants or (('fp32', 'tf32', 'tf32_fullwrite') if allocator else ('fp32', 'tf32')))
    assert variants and set(variants) <= {'fp32', 'tf32', 'tf32_fullwrite', 'tf32_layout'}
    assert allocator or not set(variants) & {'tf32_fullwrite', 'tf32_layout'}
    mm = layout = layout_direct = None
    if 'tf32_layout' in variants:
        mm = importlib.import_module('flag_gems.ops.mm')
        layout = importlib.import_module('flag_gems.ops.mm_preserve_layout')
        layout_direct = layout.make_direct(mm)
    order, audit, checkpoints, allocation_counts = [], {}, {}, {}
    model = remaining[remaining.index('--model') + 1]
    original_load = torch.load
    def load(path, *values, **kwargs):
        result = original_load(path, *values, **kwargs)
        if isinstance(path, (str, Path)) and Path(path).stem == model:
            digest = hashlib.sha256()
            with Path(path).open('rb') as handle:
                for chunk in iter(lambda: handle.read(8*1024**2), b''):
                    digest.update(chunk)
            checkpoints[str(path)] = digest.hexdigest()
        return result
    torch.load = load
    original = batch_inference.get_default_runner
    def initialize(*values, **kwargs):
        runner = original(*values, **kwargs)
        predict = runner.predict
        def run(*values, **kwargs):
            mode = variants[len(order) % len(variants)]
            order.append(mode)
            first = mode not in audit
            torch.backends.cuda.matmul.allow_tf32 = mode != 'fp32'
            torch.backends.cudnn.allow_tf32 = mode != 'fp32'
            if allocator:
                os.environ['PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION'] = 'ieee' if mode == 'fp32' else 'tf32'
                allocator.ENABLED, allocator.AUDIT, allocator.POISON = mode in ('tf32_fullwrite','tf32_layout'), first, args.poison and first
                allocator.CALLS.clear()
            original_direct = mm._direct_mm if mm else None
            if layout:
                layout.AUDIT = first
                layout.CALLS.clear()
                if mode == 'tf32_layout':
                    mm._direct_mm = layout_direct
            try:
                with audit_prediction(audit, mode) if first else nullcontext():
                    return predict(*values, **kwargs)
            finally:
                if allocator:
                    if first:
                        allocation_counts[mode] = dict(allocator.CALLS)
                    allocator.ENABLED = allocator.AUDIT = allocator.POISON = False
                if layout:
                    if first:
                        allocation_counts[mode].update(layout.CALLS)
                    layout.AUDIT = False
                    mm._direct_mm = original_direct
        runner.predict = run
        return runner
    batch_inference.get_default_runner = initialize
    sys.argv = [str(args.boundary_harness), *remaining]
    runpy.run_path(str(args.boundary_harness), run_name='__main__')
    bench = json.loads((args.paired_output.parent/'benchmark.json').read_text())
    refs = json.loads(args.reference_pair.read_text())['summary']
    assert len(order) == len(bench['records']) and checkpoints
    summary = {}
    for mode in variants:
        rows = [r for r, m in zip(bench['records'], order) if m == mode and r['phase'] == 'repeat']
        warm = [r for r, m in zip(bench['records'], order) if m == mode and r['phase'] == 'warmup']
        assert len(rows) == args.per_mode_repeats and len(warm) == args.per_mode_warmup, (mode, len(rows), len(warm))
        ref = refs['tf32' if mode in ('tf32_fullwrite','tf32_layout') else mode]
        assert all(r['prediction_digest'] == ref['prediction_digest'] and
                   r['five_confidence_metrics'] == ref['five_confidence_metrics'] for r in rows+warm), mode
        summary[mode] = {'count': len(rows), 'warmup_count': len(warm),
                         'median_seconds': statistics.median(r['wall_seconds'] for r in rows),
                         'samples_seconds': [r['wall_seconds'] for r in rows],
                         'prediction_digest': rows[0]['prediction_digest'],
                         'five_confidence_metrics': rows[0]['five_confidence_metrics'],
                         'bitwise_equal_frozen_reference': True}
    if allocator and 'tf32_fullwrite' in variants:
        assert allocation_counts['tf32_fullwrite'].get('allocations', 0) > 0
        assert not allocation_counts.get('fp32') and not allocation_counts.get('tf32')
    if layout:
        assert allocation_counts['tf32_layout'].get('prepared_calls', 0) > 0
    args.paired_output.write_text(json.dumps({'status': 'PASS', 'order': order, 'summary': summary,
        'checkpoint_sha256': checkpoints, 'allocation_counts_first_warmup': allocation_counts,
        'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'poison_first_warmup': args.poison,
        'scope': 'Inference screen; allocation-only candidate. Original FP32/TF32 hashes and all five confidence scores retained.'}, indent=2)+'\n')
    (args.paired_output.parent/'runtime_precision.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('ALLOCATION_MATRIX_PASS', model, {m: r['median_seconds'] for m, r in summary.items()}, flush=True)


if __name__ == '__main__':
    main()
