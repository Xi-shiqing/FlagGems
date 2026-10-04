"""Alternate baseline, GEMV, BMM policy TF32 and unchanged strict FP32."""
import argparse
from contextlib import nullcontext
import importlib
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
from controls import controls, switch
from runtime_precision import audit_prediction


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--harness', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--gemv-baseline', type=Path, required=True)
    p.add_argument('--paired-output', type=Path, required=True)
    p.add_argument('--per-mode-warmup', type=int, default=3)
    p.add_argument('--per-mode-repeats', type=int, default=10)
    args, remaining = p.parse_known_args()
    import torch
    from runner import batch_inference
    entries = controls(args.baseline)
    gemv_entries = controls(args.gemv_baseline)
    original_init = batch_inference.get_default_runner
    variants, order, audit = ('baseline_tf32', 'gemv_tf32', 'candidate_tf32', 'fp32'), [], {}
    rank_calls = {}
    active = [None]
    counting = [False]
    for name in ('mm',):
        module = importlib.import_module('flag_gems.ops.'+name)
        original = module.mm_gemv
        def counted(*a, _name=name, _original=original, **kw):
            value = _original(*a, **kw)
            if counting[0] and value is not None:
                rank_calls[active[0]] = rank_calls.get(active[0], 0)+1
            return value
        module.mm_gemv = counted

    def initialize(*a, **kw):
        runner = original_init(*a, **kw)
        predict = runner.predict
        def run(*a, **kw):
            variant = variants[len(order)%len(variants)]
            order.append(variant)
            active[0] = variant
            switch(gemv_entries if variant == 'gemv_tf32' else entries,
                   variant not in ('baseline_tf32','gemv_tf32'))
            torch.backends.cuda.matmul.allow_tf32 = variant != 'fp32'
            torch.backends.cudnn.allow_tf32 = variant != 'fp32'
            os.environ['PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION'] = 'ieee' if variant == 'fp32' else 'tf32'
            counting[0] = variant in ('gemv_tf32','candidate_tf32') and variant not in audit
            try:
                with audit_prediction(audit, variant) if variant not in audit else nullcontext():
                    return predict(*a, **kw)
            finally:
                counting[0] = False
        runner.predict = run
        return runner

    batch_inference.get_default_runner = initialize
    sys.argv = [str(args.harness), *remaining]
    runpy.run_path(str(args.harness), run_name='__main__')
    bench = json.loads((args.paired_output.parent/'benchmark.json').read_text())
    assert len(order)==len(bench['records'])
    summaries = {}
    for mode in variants:
        rows = [r for r,v in zip(bench['records'],order) if v==mode and r['phase']=='repeat']
        warm = [r for r,v in zip(bench['records'],order) if v==mode and r['phase']=='warmup']
        assert len(rows)==args.per_mode_repeats and len(warm)==args.per_mode_warmup
        assert len({r['prediction_digest']['sha256'] for r in rows+warm})==1,mode
        ref_mode='fp32' if mode=='fp32' else 'candidate_tf32'
        ref=json.loads(args.references.read_text())['summary'][ref_mode]
        same=rows[0]['prediction_digest']==ref['prediction_digest']
        if mode in ('baseline_tf32','fp32'):
            assert same and rows[0]['five_confidence_metrics']==ref['five_confidence_metrics'],mode
        summaries[mode]={'count':len(rows),'warmup_count':len(warm),
            'median_seconds':statistics.median(r['wall_seconds'] for r in rows),
            'samples_seconds':[r['wall_seconds'] for r in rows],
            'prediction_digest':rows[0]['prediction_digest'],
            'five_confidence_metrics':rows[0]['five_confidence_metrics'],
            'prediction_equal_reference':same}
    (args.paired_output.parent/'runtime_precision.json').write_text(json.dumps(audit,indent=2)+'\n')
    args.paired_output.write_text(json.dumps({'status':'PASS','order':order,'summary':summaries,
        'mm_gemv_calls_first_warmup':rank_calls,
        'policy':'Candidate: existing IEEE MV for vector-result MM; vector-result BMM and K<32 IEEE, remaining BMM RNE TF32. Isolated experiment; all FP32 paths unchanged.'},indent=2)+'\n')
    print('GEMV_BMM_MODEL_PASS',rank_calls,summaries,flush=True)


if __name__=='__main__':
    main()
