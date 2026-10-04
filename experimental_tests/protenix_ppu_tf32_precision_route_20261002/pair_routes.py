"""Four-mode baseline entry with bounded first-warmup arithmetic probes."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
from audit_routes import audit_routes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--route-harness', type=Path, required=True)
    p.add_argument('--route-output', type=Path, required=True)
    args, remaining = p.parse_known_args()
    extra = os.environ.get('PROTENIX_EXTRA_SITE_PACKAGES')
    if extra:
        sys.path.append(extra)
    import torch
    from runner import batch_inference
    original_init = batch_inference.get_default_runner
    variants, order, audit = ('fp32', 'tf32'), [], {}

    def initialize(*a, **kw):
        runner = original_init(*a, **kw)
        predict = runner.predict
        def run(*a, **kw):
            variant = variants[len(order) % len(variants)]
            order.append(variant)
            torch.backends.cuda.matmul.allow_tf32 = variant == 'tf32'
            torch.backends.cudnn.allow_tf32 = variant == 'tf32'
            os.environ['PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION'] = 'ieee' if variant == 'fp32' else 'tf32'
            with audit_routes(audit, variant) if variant not in audit else nullcontext():
                return predict(*a, **kw)
        runner.predict = run
        return runner

    batch_inference.get_default_runner = initialize
    sys.argv = [str(args.route_harness), *remaining]
    runpy.run_path(str(args.route_harness), run_name='__main__')
    bench = json.loads((args.route_output.parent/'benchmark.json').read_text())
    assert len(order) == len(bench['records'])
    summaries = {}
    for variant in variants:
        rows = [r for r, v in zip(bench['records'], order) if v == variant and r['phase'] == 'repeat']
        warm = [r for r, v in zip(bench['records'], order) if v == variant and r['phase'] == 'warmup']
        assert len(rows) == 10 and len(warm) == 3
        assert len({r['prediction_digest']['sha256'] for r in rows}) == 1
        summaries[variant] = {'median_seconds': statistics.median(r['wall_seconds'] for r in rows),
                             'samples_seconds': [r['wall_seconds'] for r in rows],
                             'prediction_digest': rows[0]['prediction_digest'],
                             'five_confidence_metrics': rows[0]['five_confidence_metrics'],
                             'warmup_count': len(warm), 'count': len(rows)}
    args.route_output.write_text(json.dumps({'status': 'PASS', 'order': order,
        'summary': summaries, 'routes': audit,
        'limitation': 'Replay precision is measured on local identical operands; cross-server input hashes must be checked.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
