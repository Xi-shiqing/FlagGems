"""2x2x2 precision ablation: geometric direction, noise embedding, BMM policy."""
import argparse
from contextlib import nullcontext
from functools import wraps
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
from controls import controls,switch
from runtime_precision import audit_prediction

VARIANTS=('tf32','geometry_ieee','noise_ieee','geometry_noise_ieee',
          'bmm_policy','geometry_bmm_policy','noise_bmm_policy',
          'geometry_noise_bmm_policy','fp32')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--role-harness',type=Path,required=True)
    p.add_argument('--paired-output',type=Path,required=True)
    p.add_argument('--reference-pair',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--per-mode-warmup',type=int,default=1)
    p.add_argument('--per-mode-repeats',type=int,default=3)
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    args,remaining=p.parse_known_args()
    assert '--use-flag-gems' in remaining
    import torch
    from runner import batch_inference
    entries=controls(args.baseline)
    order=[];audit={};calls={};matched={'geometry':[],'noise':[]}
    active=[None]
    first=[False]
    original=batch_inference.get_default_runner
    def restore(call,name,role):
        @wraps(call)
        def run(*a,**kw):
            enabled=active[0] is not None and role in active[0] and active[0]!='fp32'
            if not enabled:return call(*a,**kw)
            if first[0]:calls.setdefault(active[0],{})[name]=calls.setdefault(active[0],{}).get(name,0)+1
            previous=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cudnn.allow_tf32=False
                return call(*a,**kw)
            finally:
                torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=previous
        return run
    def initialize(*a,**kw):
        runner=original(*a,**kw)
        for name,module in runner.model.named_modules():
            role=attr=None
            if type(module).__name__=='AtomAttentionEncoder':role,attr='geometry','linear_no_bias_d'
            if type(module).__name__=='DiffusionConditioning':role,attr='noise','linear_no_bias_n'
            if role:
                child=getattr(module,attr)
                path=name+'.'+attr
                matched[role].append(path)
                child.forward=restore(child.forward,path,role)
        assert all(matched.values()),matched
        predict=runner.predict
        def run(*a,**kw):
            mode=args.variants[len(order)%len(args.variants)];order.append(mode);active[0]=mode
            switch(entries,'bmm_policy' in mode)
            torch.backends.cuda.matmul.allow_tf32=mode!='fp32'
            torch.backends.cudnn.allow_tf32=mode!='fp32'
            os.environ['PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION']='ieee' if mode=='fp32' else 'tf32'
            first[0]=mode not in audit
            try:
                with audit_prediction(audit,mode) if first[0] else nullcontext():
                    return predict(*a,**kw)
            finally:first[0]=False
        runner.predict=run
        return runner
    batch_inference.get_default_runner=initialize
    sys.argv=[str(args.role_harness),*remaining]
    runpy.run_path(str(args.role_harness),run_name='__main__')
    bench=json.loads((args.paired_output.parent/'benchmark.json').read_text())
    refs=json.loads(args.reference_pair.read_text())['summary']
    assert len(order)==len(bench['records'])
    summary={}
    for mode in args.variants:
        rows=[r for r,m in zip(bench['records'],order) if m==mode and r['phase']=='repeat']
        warm=[r for r,m in zip(bench['records'],order) if m==mode and r['phase']=='warmup']
        assert len(rows)==args.per_mode_repeats and len(warm)==args.per_mode_warmup
        assert len({r['prediction_digest']['sha256'] for r in rows+warm})==1,mode
        if mode in ('fp32','tf32','bmm_policy'):
            ref=refs['gemv_tf32' if mode=='tf32' else 'candidate_tf32' if mode=='bmm_policy' else 'fp32']
            assert all(r['prediction_digest']==ref['prediction_digest'] and
                       r['five_confidence_metrics']==ref['five_confidence_metrics'] for r in rows+warm),mode
        if 'geometry' in mode or 'noise' in mode:assert calls.get(mode),mode
        summary[mode]={'count':len(rows),'warmup_count':len(warm),
            'median_seconds':statistics.median(r['wall_seconds'] for r in rows),
            'samples_seconds':[r['wall_seconds'] for r in rows],
            'prediction_digest':rows[0]['prediction_digest'],
            'five_confidence_metrics':rows[0]['five_confidence_metrics']}
    (args.paired_output.parent/'runtime_precision.json').write_text(json.dumps(audit,indent=2)+'\n')
    args.paired_output.write_text(json.dumps({'status':'PASS','order':order,'summary':summary,
        'roles':matched,'restored_calls_first_warmup':calls,
        'scope':'Semantic module roles, no input or exact-shape whitelist. Complete 2x2x2 factorial or named follow-up variants. BMM policy is previous isolated experiment, not a production rule.'},indent=2)+'\n')
    print('ROLE_ABLATION_PASS',summary,flush=True)


if __name__=='__main__':main()
