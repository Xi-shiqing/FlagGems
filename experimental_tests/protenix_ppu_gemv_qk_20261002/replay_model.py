"""Instrument one TF32 warmup and assert ordinary model predictions unchanged."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import runpy
import sys
from replay_qk import replay


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--replay-harness',type=Path,required=True)
    p.add_argument('--replay-output',type=Path,required=True)
    p.add_argument('--reference-pair',type=Path,required=True)
    p.add_argument('--reference-mode',required=True)
    p.add_argument('--replay-baseline',type=Path)
    p.add_argument('--replay-kind',choices=('pairformer','vector'),default='pairformer')
    args, remaining = p.parse_known_args()
    from runner import batch_inference
    initialize = batch_inference.get_default_runner
    count = [0]
    def init(*a, **kw):
        runner = initialize(*a, **kw)
        predict = runner.predict
        def run(*a, **kw):
            count[0] += 1
            with replay(args.replay_output,args.replay_baseline,args.replay_kind) if count[0] == 1 else nullcontext():
                return predict(*a, **kw)
        runner.predict = run
        return runner
    batch_inference.get_default_runner = init
    sys.argv=[str(args.replay_harness),*remaining]
    runpy.run_path(str(args.replay_harness),run_name='__main__')
    benchmark=json.loads((args.replay_output.parent/'benchmark.json').read_text())
    ref=json.loads(args.reference_pair.read_text())['summary'][args.reference_mode]
    assert all(r['prediction_digest']==ref['prediction_digest'] and
               r['five_confidence_metrics']==ref['five_confidence_metrics'] for r in benchmark['records'])
    (args.replay_output.parent/'teacher_integrity.json').write_text(json.dumps({'status':'PASS',
        'prediction_and_five_confidence_equal_reference':True,
        'prediction_digest':ref['prediction_digest'],'prediction_count':count[0]},indent=2)+'\n')
    print('REPLAY_TEACHER_INTEGRITY_PASS',flush=True)


if __name__=='__main__':main()
