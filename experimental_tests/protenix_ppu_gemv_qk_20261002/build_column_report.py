"""Join all precision modes with matched hashes and explicit experimental limits."""
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT=Path(__file__).resolve().parent
METRICS=('plddt','gpde','ptm','iptm','ranking_score')
MODES=('h100_fp32','h100_tf32','ppu_fp32','ppu_tf32','ppu_candidate_tf32')


def read(path):
    return json.loads(path.read_text())


def write_csv(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gain(before,after):
    assert len(before)==len(after)==10
    rng=random.Random(101)
    values=[]
    for _ in range(5000):
        idx=[rng.randrange(len(before)) for _ in before]
        values.append(100*(1-statistics.median(after[i] for i in idx)/
                           statistics.median(before[i] for i in idx)))
    values.sort()
    return {'latency_reduction_percent':100*(1-statistics.median(after)/statistics.median(before)),
            'paired_bootstrap_ci95_percent':[values[124],values[4874]]}


def main():
    op_times,op_errors,gains=[],[],[]
    p=read(ROOT/'ppu/column_operators.json');h=read(ROOT/'h100/operators.json')
    assert p['status']==h['status']=='PASS'
    for pp,hh in zip(p['cases'],h['cases']):
        assert pp['name']==hh['name'] and pp['shape']==hh['shape']
        assert pp['input_sha256']==hh['input_sha256'] and pp['gold_sha256']==hh['gold_sha256']
        assert pp['candidate_fp32_bitwise_equal_baseline']
        values={**{'h100_'+m:v for m,v in hh['modes'].items()},
                **{'ppu_'+m:v for m,v in pp['modes'].items()}}
        row={'operator':pp['name'],'shape':str(pp['shape']),'matched_inputs':True}
        for m in MODES:
            for phase in ('inference','training'):
                row[m+'_'+phase+'_ms']=values[m]['median_'+phase+'_ms']
        op_times.append(row)
        for field in pp['gold_sha256']:
            row={'operator':pp['name'],'shape':str(pp['shape']),'field':field,
                 'reference':'Full CPU FP64 gold; updated parameters use one-rounding FP32 SGD'}
            for m in MODES:
                for metric in ('relative_l2','max_abs','sha256'):
                    row[m+'_'+metric]=values[m]['errors_vs_cpu_fp64'][field][metric]
            op_errors.append(row)
        for phase in ('inference','training'):
            gains.append({'operator':pp['name'],'phase':phase,
                **gain(pp['modes']['tf32']['samples_ms'][phase+'_ms'],
                       pp['modes']['candidate_tf32']['samples_ms'][phase+'_ms'])})

    bmm_times,bmm_errors,bmm_cross=[],[],[]
    p,h=[read(ROOT/s/'bmm_policy.json') for s in ('ppu','h100')]
    assert p['status']==h['status']=='PASS'
    for pp,hh in zip(p['cases'],h['cases']):
        assert pp['name']==hh['name'] and pp['inputs']==hh['inputs'] and pp['gold']==hh['gold']
        assert pp['candidate_fp32_bitwise_equal_baseline']
        values={**{'h100_'+m:v for m,v in hh['modes'].items()},
                **{'ppu_'+m:v for m,v in pp['modes'].items()}}
        row={'case':pp['name'],'shape':str(pp['shape']),'matched_inputs':True}
        for m in MODES:
            row[m+'_forward_ms']=values[m]['median_inference_ms']
        bmm_times.append(row)
        for field in pp['gold']:
            row={'case':pp['name'],'shape':str(pp['shape']),'field':field}
            for m in MODES:
                for metric in ('relative_l2','max_abs','sha256'):
                    row[m+'_'+metric]=values[m]['errors_vs_cpu_fp64'][field][metric]
            bmm_errors.append(row)
        for mode in ('ppu_tf32','ppu_candidate_tf32'):
            # Use the tighter of IEEE/RNE gold-based triangle inequalities.
            intervals=[]
            for ref in ('ieee','rne'):
                e=[]
                for m in (mode,'h100_tf32'):
                    v=values[m]
                    e.append(v['errors_vs_cpu_fp64']['output']['relative_l2'] if ref=='ieee'
                             else v['forward_vs_rne_fp64']['relative_l2'])
                intervals.append((max(0,abs(e[0]-e[1]))/(1+e[1]),
                                  (e[0]+e[1])/max(1e-30,1-e[1])))
            equal=values[mode]['errors_vs_cpu_fp64']['output']['sha256']==values['h100_tf32']['errors_vs_cpu_fp64']['output']['sha256']
            bmm_cross.append({'case':pp['name'],'shape':str(pp['shape']),'mode':mode,
                'reference':'h100_tf32','forward_bitwise_equal':equal,
                'relative_l2_lower_bound':0 if equal else max(x[0] for x in intervals),
                'relative_l2_upper_bound':0 if equal else min(x[1] for x in intervals),
                'method':'Matched inputs; equality by full output hash, otherwise rigorous IEEE/RNE gold norm bounds. No raw cross-server tensor export.'})

    live=[]
    p=read(ROOT/'ppu/qk_candidate/G02/replay.json')
    h=read(ROOT/'h100/qk/G02/replay.json')
    assert read(ROOT/'ppu/qk_candidate/G02/teacher_integrity.json')['status']=='PASS'
    for pp,hh in zip(p['records'],h['records']):
        assert pp['role']==hh['role']
        row={'role':pp['role'],'a_shape':str(pp['inputs'][0]['shape']),
             'b_shape':str(pp['inputs'][1]['shape']),
             'cross_device_input_hashes_equal':all(a['sha256']==b['sha256'] for a,b in zip(pp['inputs'],hh['inputs'])),
             'scope':'Each device uses its own actual TF32 model activations; same operands across precision replays on that device'}
        values={**{'h100_'+m:v for m,v in hh['modes'].items()},
                **{'ppu_'+m:v for m,v in pp['modes'].items()}}
        for m in MODES:
            row[m+'_ms']=values[m]['median_ms']
            row[m+'_relative_l2_vs_own_ieee_gold']=values[m]['vs_ieee_fp64']['relative_l2']
            row[m+'_relative_l2_vs_own_rne_gold']=values[m]['vs_rne_fp64']['relative_l2']
        live.append(row)

    p=read(ROOT/'ppu/column_derived.json');h=read(ROOT/'h100/derived.json')
    assert p['checkpoint']['sha256']==h['checkpoint']['sha256']
    pp,hh=p['cases']['G02'],h['cases']['G02']
    for key in ('canonical_input_sha256','asset_hashes','atom_count','atom_identity_sha256'):
        assert pp['input_audit'][key]==hh['input_audit'][key],key
    assert pp['input_audit']['features'].keys()==hh['input_audit']['features'].keys()
    for key in pp['input_audit']['features']:
        a,b=pp['input_audit']['features'][key],hh['input_audit']['features'][key]
        assert (a['sha256'],a['shape'],a['dtype'])==(b['sha256'],b['shape'],b['dtype']),key
    for key in ('initialization_semantic_sha256','prediction_semantic_sha256'):
        assert pp['rng_audit'][key]==hh['rng_audit'][key]
    prior=ROOT.parent/'rank_one_20261002'
    for s,case in (('ppu',pp),('h100',hh)):
        old=read(prior/s/'derived.json')['cases']['G02']['summary']
        for mode in ('fp32','tf32') if s=='h100' else ('fp32','baseline_tf32'):
            ref=mode if s=='h100' or mode=='fp32' else 'candidate_tf32'
            assert case['summary'][mode]['prediction_digest']==old[ref]['prediction_digest']
    values={**{'h100_'+m:v for m,v in hh['summary'].items()},
            **{'ppu_'+('tf32' if m=='baseline_tf32' else m):v for m,v in pp['summary'].items()}}
    inference={'model':'protenix_base_default_v1.0.0','input':'G02','N_token':pp['input_audit']['N_token'],
               'cycle':1,'step':1,'sample':1,'warmup_per_mode':3,'repeats_per_mode':10}
    for m,v in values.items():
        inference[m+'_seconds']=v['median_seconds']
    confidence=[]
    for metric in METRICS:
        confidence.append({'metric':metric,**{m:v['five_confidence_metrics'][metric] for m,v in values.items()}})
    anchors=read(prior/'previous_matched_anchors.json')
    def anchor(mode,ref):
        return next(r for r in anchors['structure'] if r['input']=='G02' and r['mode']==mode and r['reference']==ref)
    fp_anchor=anchor('ppu_fp32','h100_fp32')
    tf_anchor=anchor('ppu_tf32','h100_tf32')
    assert hh['drift_vs_frozen']['tf32']['rmsd_vs_frozen_same_precision_A']<1e-10
    structure=[{'mode':'ppu_fp32','reference':'h100_fp32',
        'rmsd_lower_bound_A':fp_anchor['rmsd_lower_bound_A'],
        'rmsd_upper_bound_A':fp_anchor['rmsd_upper_bound_A'],
        'method':'Frozen strict FP32 anchor; both current output hashes unchanged'}]
    def distance(a,b):
        return pp['same_device_rmsd_A'].get(a+'_vs_'+b,pp['same_device_rmsd_A'].get(b+'_vs_'+a))
    for mode in ('baseline_tf32','gemv_tf32','candidate_tf32'):
        d=pp['drift_vs_frozen'][mode]['rmsd_vs_frozen_same_precision_A']
        lo=max(0,tf_anchor['rmsd_exact_A']-d);hi=tf_anchor['rmsd_exact_A']+d
        c=distance(mode,'fp32');t=hh['same_device_rmsd_A']['fp32_vs_tf32'];e=fp_anchor['rmsd_upper_bound_A']
        lo=max(lo,abs(c-t)-e,0);hi=min(hi,c+t+e)
        structure.append({'mode':'ppu_'+('tf32' if mode=='baseline_tf32' else mode),'reference':'h100_tf32',
            'rmsd_lower_bound_A':lo,'rmsd_upper_bound_A':hi,
            'method':'Rigorous intersection of TF32 and strict FP32 anchor triangle inequalities, not direct cross-server RMSD'})
    for row in structure:
        row['rmsd_gate_0_1A']='PASS' if row['rmsd_upper_bound_A']<=.1 else 'FAIL' if row['rmsd_lower_bound_A']>.1 else 'UNRESOLVED'
        metric_rows=[]
        for metric in METRICS:
            a,b=[values[m]['five_confidence_metrics'][metric] for m in (row['mode'],row['reference'])]
            relative=100*abs(a-b)/abs(b) if b else None
            metric_rows.append({'metric':metric,'absolute_delta':a-b,'relative_percent':relative,
                                'pass_5pct':relative<=5 if relative is not None else a==b})
        row['confidence_comparison']=metric_rows
        row['five_confidence_gate_5pct']=all(x['pass_5pct'] for x in metric_rows)
    inventories={s:read(ROOT/s/'source_inventory.json') for s in ('ppu','h100')}
    assert inventories['ppu']['sources']['gemv/mm_gemv']==hashlib.sha256((ROOT/'mm_gemv.py').read_bytes()).hexdigest()
    protected=ROOT.parent.parent/'ppu_mean_fix_20260920/benchmark_e2e.py'
    assert hashlib.sha256(protected.read_bytes()).hexdigest()=='6ede90f98d9435b8509fb37b086c6a490296b49a99c6fbbd4094eda2f5506d42'
    rejected=read(ROOT/'complete_comparison_broad.json')
    vector=[]
    pv,hv=[read(ROOT/server/'vector_mm/G02/replay.json')['records'][0] for server in ('ppu','h100')]
    assert all(a['sha256']==b['sha256'] for a,b in zip(pv['inputs'],hv['inputs']))
    vv={**{'h100_'+m:v for m,v in hv['modes'].items()},**{'ppu_'+m:v for m,v in pv['modes'].items()}}
    vector_row={'op':'diffusion_vector_mm','a_shape':str(pv['inputs'][0]['shape']),'b_shape':str(pv['inputs'][1]['shape']),
                'input_hashes_equal':True,'scope':'Actual activations; broad row MV rejected for model RMSD regression. CPU IEEE gold accumulation differs between hosts; errors use each host gold.'}
    for mode in MODES:
        vector_row[mode+'_ms']=vv[mode]['median_ms']
        vector_row[mode+'_relative_l2_vs_own_ieee_gold']=vv[mode]['vs_ieee_fp64']['relative_l2']
    vector.append(vector_row)
    # Give each candidate a distinct name in the user-facing joined tables.
    def rename(rows, mapping):
        result=[]
        for row in rows:
            value={}
            for key,item in row.items():
                for old,new in mapping.items():
                    key=key.replace(old,new)
                if key in ('mode','reference') and isinstance(item,str):
                    item=mapping.get(item,item)
                value[key]=item
            result.append(value)
        return result
    column={'ppu_candidate_tf32':'ppu_column_mm_tf32'}
    bmm={'ppu_candidate_tf32':'ppu_bmm_policy_tf32'}
    model={'ppu_candidate_tf32':'ppu_column_mm_bmm_policy_tf32','ppu_gemv_tf32':'ppu_column_mm_tf32'}
    op_times=rename(op_times,column);op_errors=rename(op_errors,column)
    bmm_times=rename(bmm_times,bmm);bmm_errors=rename(bmm_errors,bmm);bmm_cross=rename(bmm_cross,bmm)
    live=rename(live,bmm)
    vector=rename(vector,{'ppu_candidate_tf32':'ppu_row_mv_tf32_rejected'})
    inference=rename([inference],model)[0];confidence=rename(confidence,model);structure=rename(structure,model)
    report={'status':'VERIFIED_GEMV_REPAIR_AND_EXPERIMENTAL_BMM_POLICY',
        'mode_definitions':{'h100_fp32':'Native H100, matmul and cuDNN TF32 disabled',
         'h100_tf32':'Native H100 with TF32 allowed; some native operations still execute IEEE',
         'ppu_fp32':'PPU plus FlagGems, matmul/cuDNN TF32 disabled and TriAttention IEEE',
         'ppu_tf32':'PPU plus FlagGems baseline TF32, includes prior K=1 repair',
         'ppu_column_mm_tf32':'Baseline plus generic N=1 MM to existing IEEE MV; no exact-shape whitelist',
         'ppu_bmm_policy_tf32':'Experimental generic BMM precision selector; model RMSD FAIL, not promoted',
         'ppu_column_mm_bmm_policy_tf32':'Column MM plus experimental BMM selector; model RMSD FAIL',
         'ppu_row_mv_tf32_rejected':'Broad row-vector MV test only; improves individual MM error but worsens model RMSD',
         'rejected_broad_forward_candidate':'Archived: ppu_gemv_tf32 includes row and column MV; ppu_candidate_tf32 also includes BMM selector'},
        'operator_timings':op_times,'operator_errors':op_errors,'operator_gains':gains,
        'bmm_timings':bmm_times,'bmm_errors':bmm_errors,'bmm_cross_device_bounds':bmm_cross,
        'actual_diffusion_vector_replay_rejected':vector,'actual_pairformer_replay':live,'inference':inference,'confidence':confidence,'structure':structure,
        'same_server_rmsd_A':{'ppu':pp['same_device_rmsd_A'],'h100':hh['same_device_rmsd_A']},
        'rejected_broad_forward_candidate':{'inference':rejected['inference'],'structure':rejected['structure'],'confidence':rejected['confidence']},
        'checkpoint':p['checkpoint'],'input_audit':pp['input_audit'],
        'rng_audit':{'ppu':pp['rng_audit'],'h100':hh['rng_audit']},
        'mm_gemv_calls_first_warmup':read(ROOT/'ppu/column_model/G02/paired.json')['mm_gemv_calls_first_warmup'],
        'source_inventory':inventories,'gemv_entry_check':read(ROOT/'ppu/check.json'),
        'limits':['Final GEMV rule is column-result MM only (N=1). Broad row-result GEMV not promoted: H100 actual diffusion row MM uses IEEE and MV lowers standalone error, but G02 model RMSD worsens. Coupled upstream error remains unresolved.',
                  'Full model training, nine-model acceptance and default long diffusion schedule not run.',
                  'BMM rule is experimental: adjacent dimensions show native H100 TF32 can still select IEEE. No claim of exact dispatch emulation.',
                  'Standalone training is forward + generated Torch backward + one SGD update; model inference timing covers predict only.',
                  'H100 GPU0 timing contaminated by another task and excluded; current timing uses idle GPU7 on original server.',
                  'No model MFU claim from these timings; production/default server package unchanged.',
                  'Raw activation file capture rejected by auto-review; live in-memory replay persisted only scalar/hash JSON.'],
        'source_validation':'GEMV helper and frozen-server actual entries GPU-tested; newer operator source tree integration syntax-checked, not full GPU-loaded.'}
    (ROOT/'complete_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,rows in (('four_mode_operator_timings.csv',op_times),('four_mode_operator_errors.csv',op_errors),
                      ('operator_gains.csv',gains),('four_mode_bmm_timings.csv',bmm_times),
                      ('four_mode_bmm_errors.csv',bmm_errors),('bmm_cross_device_bounds.csv',bmm_cross),
                      ('four_mode_actual_pairformer.csv',live),('four_mode_actual_diffusion_vector_rejected.csv',vector),('four_mode_inference.csv',[inference]),
                      ('four_mode_confidence.csv',confidence),('four_mode_structure.csv',[
                          {k:v for k,v in row.items() if k!='confidence_comparison'} for row in structure])):
        write_csv(name,rows)
    print('REPORT_HASH_RNG_INPUT_CHECKPOINT_PASS',inference,structure)


if __name__=='__main__':
    main()
