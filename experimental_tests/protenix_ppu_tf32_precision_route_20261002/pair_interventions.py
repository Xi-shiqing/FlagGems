"""Generic operator-family precision controls, without input/shape dispatch rules."""
import argparse
from contextlib import nullcontext
import importlib
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
from runtime_precision import audit_prediction


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--route-harness', type=Path, required=True)
    p.add_argument('--route-output', type=Path, required=True)
    args, remaining = p.parse_known_args()
    assert '--use-flag-gems' in remaining
    extra = os.environ.get('PROTENIX_EXTRA_SITE_PACKAGES')
    if extra:
        sys.path.append(extra)
    import torch
    import torch.nn.functional as F
    from runner import batch_inference
    bmm = importlib.import_module('flag_gems.ops.bmm')
    original_bmm_policy = bmm._allow_bmm_tf32
    original_sdpa = F.scaled_dot_product_attention
    original_init = batch_inference.get_default_runner
    variants = ('tf32', 'bmm_ieee', 'sdpa_ieee', 'bmm_sdpa_ieee', 'fp32')
    order, audit, sdpa_counts = [], {}, {}
    active = [None]
    counting = [False]

    def bmm_ieee(*args):
        return False

    def sdpa_ieee(*a, **kw):
        if counting[0]:
            sdpa_counts[active[0]] = sdpa_counts.get(active[0], 0) + 1
        previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            return original_sdpa(*a, **kw)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous

    def initialize(*a, **kw):
        runner = original_init(*a, **kw)
        predict = runner.predict
        def run(*a, **kw):
            variant = variants[len(order) % len(variants)]
            order.append(variant)
            active[0] = variant
            bmm._allow_bmm_tf32 = bmm_ieee if variant in ('bmm_ieee', 'bmm_sdpa_ieee') else original_bmm_policy
            F.scaled_dot_product_attention = sdpa_ieee if variant in ('sdpa_ieee', 'bmm_sdpa_ieee') else original_sdpa
            torch.backends.cuda.matmul.allow_tf32 = variant != 'fp32'
            torch.backends.cudnn.allow_tf32 = variant != 'fp32'
            os.environ['PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION'] = 'ieee' if variant == 'fp32' else 'tf32'
            counting[0] = variant not in audit
            try:
                with audit_prediction(audit, variant) if counting[0] else nullcontext():
                    return predict(*a, **kw)
            finally:
                counting[0] = False
                bmm._allow_bmm_tf32 = original_bmm_policy
                F.scaled_dot_product_attention = original_sdpa
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
        assert len({r['prediction_digest']['sha256'] for r in rows}) == 1, variant
        summaries[variant] = {'median_seconds': statistics.median(r['wall_seconds'] for r in rows),
            'samples_seconds': [r['wall_seconds'] for r in rows],
            'prediction_digest': rows[0]['prediction_digest'],
            'five_confidence_metrics': rows[0]['five_confidence_metrics'],
            'warmup_count': len(warm), 'count': len(rows)}
    args.route_output.write_text(json.dumps({'status': 'PASS', 'order': order,
        'summary': summaries, 'runtime_precision': audit, 'sdpa_override_first_warmup_calls': sdpa_counts,
        'policy': 'BMM override applies to the whole registered operator family; SDPA override applies to the functional entry, including native Efficient SDPA and Math decomposition.',
        'limitation': 'Precision interventions are separate diagnostics, not ordinary TF32 or an accepted production candidate.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
