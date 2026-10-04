"""Alternate native H100 FP32/TF32 in one initialized model."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import runpy
import statistics
import sys
from runtime_precision import audit_prediction


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native-harness', dest='harness', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--paired-output', type=Path, required=True)
    args, remaining = p.parse_known_args()
    assert '--use-flag-gems' not in remaining
    import torch
    from runner import batch_inference
    original_init = batch_inference.get_default_runner
    variants = ('fp32', 'tf32')
    order, audit = [], {}

    def initialize(*a, **kw):
        runner = original_init(*a, **kw)
        original_predict = runner.predict
        def predict(*a, **kw):
            variant = variants[len(order) % 2]
            order.append(variant)
            torch.backends.cuda.matmul.allow_tf32 = variant == 'tf32'
            torch.backends.cudnn.allow_tf32 = variant == 'tf32'
            with audit_prediction(audit, variant) if variant not in audit else nullcontext():
                return original_predict(*a, **kw)
        runner.predict = predict
        return runner

    batch_inference.get_default_runner = initialize
    sys.argv = [str(args.harness), *remaining]
    runpy.run_path(str(args.harness), run_name='__main__')
    bench = json.loads((args.paired_output.parent / 'benchmark.json').read_text())
    assert not bench['environment'].get('use_flag_gems', False)
    assert len(order) == len(bench['records'])
    summaries = {}
    for variant in variants:
        rows = [r for r, v in zip(bench['records'], order) if v == variant and r['phase'] == 'repeat']
        warm = [r for r, v in zip(bench['records'], order) if v == variant and r['phase'] == 'warmup']
        assert len(rows) == 10 and len(warm) == 3
        ref = next(r for r in json.loads((args.references / variant / 'benchmark.json').read_text())['records'] if r['phase'] == 'repeat')
        assert len({r['prediction_digest']['sha256'] for r in rows}) == 1, variant
        values = [r['wall_seconds'] for r in rows]
        summaries[variant] = {'count': len(rows), 'warmup_count': len(warm),
            'median_seconds': statistics.median(values), 'samples_seconds': values,
            'prediction_equal_reference': all(r['prediction_digest'] == ref['prediction_digest'] for r in rows),
            'confidence_equal_reference': all(r['five_confidence_metrics'] == ref['five_confidence_metrics'] for r in rows),
            'prediction_digest': rows[0]['prediction_digest'],
            'five_confidence_metrics': rows[0]['five_confidence_metrics']}
    (args.paired_output.parent / 'runtime_precision.json').write_text(json.dumps(audit, indent=2) + '\n')
    args.paired_output.write_text(json.dumps({'status': 'PASS', 'native_h100': True,
        'order': order, 'summary': summaries,
        'scope': 'InferenceRunner.predict; 3 warmups and 10 measured predictions per precision'}, indent=2) + '\n')
    print('MATCHED_NATIVE_PASS', summaries, flush=True)


if __name__ == '__main__':
    main()
