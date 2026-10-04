"""Four-mode boundary evidence on original devices, with unchanged predictions."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
from replay_forward_trial import TriangleReplay
from runtime_precision import audit_prediction


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--boundary-harness',type=Path,required=True)
    p.add_argument('--paired-output',type=Path,required=True)
    p.add_argument('--reference-pair',type=Path,required=True)
    p.add_argument('--server',choices=('ppu','h100'),required=True)
    p.add_argument('--per-mode-warmup',type=int,default=3)
    p.add_argument('--per-mode-repeats',type=int,default=10)
    args,remaining=p.parse_known_args()
    import torch
    from runner import batch_inference
    original=batch_inference.get_default_runner
    order=[];audit={};tracer=[None]
    variants=('fp32','tf32','forward_kv_tf32') if args.server=='ppu' else ('fp32','tf32')
    candidate_calls=[]
    def initialize(*a,**kw):
        runner=original(*a,**kw)
        tracer[0]=TriangleReplay(runner.model)
        public_triangle=tracer[0].original
        def candidate_triangle(*a,**kw):
            candidate_calls.append({'shape':list(a[0].shape),'precision':kw.get('precision')})
            return tracer[0].candidates['forward_kv_prequant'].triangle_attention(*a,**kw)
        print('BOUNDARY_SELECTION',len(tracer[0].selected),flush=True)
        predict=runner.predict
        def run(*a,**kw):
            mode=variants[len(order)%len(variants)];order.append(mode)
            torch.backends.cuda.matmul.allow_tf32=mode!='fp32'
            torch.backends.cudnn.allow_tf32=mode!='fp32'
            if args.server=='ppu':
                os.environ['PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION']='ieee' if mode=='fp32' else 'tf32'
            first=mode not in audit
            trace_first=first and mode in ('fp32','tf32')
            if trace_first:tracer[0].begin(mode)
            if mode=='forward_kv_tf32':tracer[0].namespace.triangle_attention=candidate_triangle
            try:
                with audit_prediction(audit,mode) if first else nullcontext():
                    return predict(*a,**kw)
            finally:
                if mode=='forward_kv_tf32':tracer[0].namespace.triangle_attention=public_triangle
                if trace_first:
                    tracer[0].end()
                    print('BOUNDARY_DONE',mode,len(tracer[0].rows[mode]),flush=True)
        runner.predict=run
        return runner
    batch_inference.get_default_runner=initialize
    sys.argv=[str(args.boundary_harness),*remaining]
    runpy.run_path(str(args.boundary_harness),run_name='__main__')
    bench=json.loads((args.paired_output.parent/'benchmark.json').read_text())
    refs=json.loads(args.reference_pair.read_text())['summary']
    assert len(order)==len(bench['records'])
    summary={}
    for mode in variants:
        rows=[r for r,m in zip(bench['records'],order) if m==mode and r['phase']=='repeat']
        warm=[r for r,m in zip(bench['records'],order) if m==mode and r['phase']=='warmup']
        assert len(rows)==args.per_mode_repeats and len(warm)==args.per_mode_warmup
        key='gemv_tf32' if args.server=='ppu' and mode!='fp32' else mode
        ref=refs[key]
        assert all(r['prediction_digest']==ref['prediction_digest'] and
                   r['five_confidence_metrics']==ref['five_confidence_metrics'] for r in rows+warm),mode
        summary[mode]={'count':len(rows),'warmup_count':len(warm),
            'median_seconds':statistics.median(r['wall_seconds'] for r in rows),
            'samples_seconds':[r['wall_seconds'] for r in rows],
            'prediction_digest':rows[0]['prediction_digest'],
            'five_confidence_metrics':rows[0]['five_confidence_metrics'],
            'prediction_and_confidence_equal_reference':True}
    (args.paired_output.parent/'triangle_rounding.json').write_text(json.dumps(tracer[0].close(),indent=2)+'\n')
    (args.paired_output.parent/'runtime_precision.json').write_text(json.dumps(audit,indent=2)+'\n')
    args.paired_output.write_text(json.dumps({'status':'PASS','order':order,'summary':summary,
        'candidate_calls':candidate_calls,'scope':f'{args.per_mode_warmup} warmups and {args.per_mode_repeats} repeats per mode; first warmup boundary trace only; baseline replay hooks removed before repeats; forward candidate routes actual PPU predictions for every candidate call.'},indent=2)+'\n')
    assert args.server!='ppu' or (candidate_calls and all(x['precision']=='tf32' for x in candidate_calls))
    print('FORWARD_MODEL_INTEGRITY_PASS',summary,flush=True)


if __name__=='__main__':main()
