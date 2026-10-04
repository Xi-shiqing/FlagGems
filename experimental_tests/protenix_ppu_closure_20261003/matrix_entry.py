"""Compare original four modes and the combined candidate in real predictions."""
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
    parser.add_argument('--per-mode-warmup', type=int, default=1)
    parser.add_argument('--per-mode-repeats', type=int, default=3)
    parser.add_argument('--paired-output', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    import torch
    from runner import batch_inference
    variants = ('fp32', 'tf32', 'combined_tf32') if args.server == 'ppu' else ('fp32', 'tf32')
    order, audit, calls, checkpoints = [], {}, Counter(), {}
    original = batch_inference.get_default_runner
    original_load = torch.load
    model = remaining[remaining.index('--model') + 1]

    def load(path, *a, **kw):
        result = original_load(path, *a, **kw)
        if isinstance(path, (str, Path)) and Path(path).stem == model:
            digest = hashlib.sha256()
            with Path(path).open('rb') as handle:
                for chunk in iter(lambda: handle.read(8*1024**2), b''):
                    digest.update(chunk)
            checkpoints[str(path)] = digest.hexdigest()
        return result
    torch.load = load

    def initialize(*a, **kw):
        runner = original(*a, **kw)
        predict = runner.predict
        candidate = namespace = None
        if args.server == 'ppu':
            namespace = importlib.import_module('flag_gems.ops')
            public = namespace.triangle_attention
            candidate = importlib.import_module('combined_64m')
            def operator(*a, **kw):
                calls[str(tuple(a[0].shape)) + ':' + str(kw.get('precision'))] += 1
                return candidate.triangle_attention(*a, **kw)
        def run(*a, **kw):
            mode = variants[len(order) % len(variants)]
            order.append(mode)
            enabled = mode != 'fp32'
            torch.backends.cuda.matmul.allow_tf32 = enabled
            torch.backends.cudnn.allow_tf32 = enabled
            if args.server == 'ppu':
                os.environ['PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION'] = 'tf32' if enabled else 'ieee'
                namespace.triangle_attention = operator if mode == 'combined_tf32' else public
            with audit_prediction(audit, mode) if mode not in audit else nullcontext():
                return predict(*a, **kw)
        runner.predict = run
        return runner
    batch_inference.get_default_runner = initialize
    sys.argv = [str(args.boundary_harness), *remaining]
    runpy.run_path(str(args.boundary_harness), run_name='__main__')
    bench = json.loads((args.paired_output.parent / 'benchmark.json').read_text())
    assert len(order) == len(bench['records'])
    summary = {}
    for mode in variants:
        rows = [r for r, m in zip(bench['records'], order) if m == mode and r['phase'] == 'repeat']
        warm = [r for r, m in zip(bench['records'], order) if m == mode and r['phase'] == 'warmup']
        assert len(rows) == args.per_mode_repeats and len(warm) == args.per_mode_warmup
        assert all(r['prediction_digest'] == rows[0]['prediction_digest'] and r['five_confidence_metrics'] == rows[0]['five_confidence_metrics'] for r in rows + warm)
        summary[mode] = {'count': len(rows), 'warmup_count': len(warm), 'samples_seconds': [r['wall_seconds'] for r in rows],
                         'median_seconds': statistics.median(r['wall_seconds'] for r in rows),
                         'prediction_digest': rows[0]['prediction_digest'], 'five_confidence_metrics': rows[0]['five_confidence_metrics']}
    if args.server == 'ppu':
        assert calls and summary['combined_tf32']['prediction_digest'] == summary['tf32']['prediction_digest']
        assert summary['combined_tf32']['five_confidence_metrics'] == summary['tf32']['five_confidence_metrics']
    assert checkpoints, 'actual checkpoint load was not audited'
    args.paired_output.write_text(json.dumps({'status': 'PASS', 'order': order, 'summary': summary, 'candidate_calls': dict(calls),
        'checkpoint_sha256': checkpoints, 'scope': 'Actual candidate predictions, matched current baseline. 1 warmup/3 repeats per mode screen; not default cycle/diffusion schedule performance acceptance.'}, indent=2) + '\n')
    (args.paired_output.parent / 'runtime_precision.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('MATRIX_PREDICTION_INTEGRITY_PASS', model, {m: v['median_seconds'] for m, v in summary.items()}, flush=True)


if __name__ == '__main__':
    main()
