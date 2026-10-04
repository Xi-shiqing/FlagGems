"""Join scalar four-mode evidence; keep formal gates distinct from equality checks."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parent
METRICS = ('plddt', 'gpde', 'ptm', 'iptm', 'ranking_score')


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_ci(old, new):
    assert len(old) == len(new)
    rng = random.Random(101)
    values = []
    for _ in range(10000):
        indices = rng.choices(range(len(old)), k=len(old))
        a = statistics.median(old[i] for i in indices)
        b = statistics.median(new[i] for i in indices)
        values.append(100*(1-b/a))
    values.sort()
    return [values[250], values[9749]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='matrix')
    parser.add_argument('--expected', type=int, default=18)
    args = parser.parse_args()
    ppu = read(ROOT/'ppu'/('derived_'+args.stage+'.json'))
    hstage = args.stage if args.stage == 'matrix' else 'screen'
    h100 = read(ROOT/'h100'/('derived_'+hstage+'.json'))
    assert ppu['status'] == h100['status'] == 'PASS'
    assert len(ppu['cases']) == len(h100['cases']) == args.expected
    timing_sources = {}
    for server, dataset in [('ppu', ppu), ('h100', h100)]:
        for row in dataset['cases']:
            timing_sources[server,row['case'],row['model']] = 'matrix (1 warmup, 5 clean repeats)' if args.stage=='matrix' else 'screen'
        recheck = ROOT/server/'derived_recheck.json'
        steady = ROOT/server/'derived_steady.json'
        if server == 'h100' and steady.exists():
            recheck = steady
        if args.stage=='matrix' and recheck.exists():
            extra=read(recheck)
            assert extra['status']=='PASS' and len(extra['cases']) == (18 if recheck == steady else 3 if server == 'ppu' else 2)
            replacements={(r['case'],r['model']):r for r in extra['cases']}
            for i, old in enumerate(dataset['cases']):
                key=(old['case'],old['model'])
                if key not in replacements:continue
                new=replacements[key]
                for field in ('feature_tensor_manifest_sha256','atom_identity_sha256','canonical_input_sha256','checkpoint_sha256','asset_hashes','initialization_rng_sha256','prediction_rng_sha256'):
                    assert old[field]==new[field], (server,key,field)
                for mode, row in old['summary'].items():
                    for field in ('prediction_digest','five_confidence_metrics'):
                        assert row[field]==new['summary'][mode][field], (server,key,mode,field)
                dataset['cases'][i]=new
                sample=next(iter(new['summary'].values()))
                timing_sources[server,*key]=f"{recheck.stem} ({sample['warmup_count']} warmups, {sample['count']} clean repeats); first run retained in derived_matrix.json"
    hp = {(r['case'], r['model']): r for r in h100['cases']}
    formal_root = ROOT.parent/'native_reference_e2e_20261004'
    if not formal_root.exists():
        formal_root = ROOT.parent/'protenix_ppu_native_reference_e2e_20261004'
    formal = read(formal_root/'summary.json')
    fp = {(r['input'], r['model']): r for r in formal['cases']}
    times, confidence, accuracy = [], [], []
    candidate_modes = set()
    for p in ppu['cases']:
        key = (p['case'], p['model'])
        h, frozen = hp[key], fp[key]
        for field in ('feature_tensor_manifest_sha256', 'atom_identity_sha256',
                      'canonical_input_sha256', 'initialization_rng_sha256', 'prediction_rng_sha256'):
            assert p[field] == h[field], (key, field)
        assert p['checkpoint_sha256'] == h['checkpoint_sha256'] and p['asset_hashes'] == h['asset_hashes']
        assert p['input_and_rng_audit'] == h['input_and_rng_audit'] == 'PASS'
        modes = [m for m in p['summary'] if m not in ('fp32', 'tf32')]
        assert modes
        candidate_modes.update(modes)
        row = {'input':p['case'], 'tokens':p['N_token'], 'model':p['model'],
               'H100_native_FP32_s':h['summary']['fp32']['median_seconds'],
               'H100_native_TF32_s':h['summary']['tf32']['median_seconds'],
               'PPU_FlagGems_FP32_s':p['summary']['fp32']['median_seconds'],
               'PPU_FlagGems_TF32_s':p['summary']['tf32']['median_seconds']}
        for mode in modes:
            old, new = p['summary']['tf32'], p['summary'][mode]
            assert old['prediction_digest'] == new['prediction_digest']
            assert old['five_confidence_metrics'] == new['five_confidence_metrics']
            row['PPU_'+mode+'_s'] = new['median_seconds']
            row[mode+'_seconds_saved'] = old['median_seconds']-new['median_seconds']
            row[mode+'_time_reduction_pct'] = 100*(1-new['median_seconds']/old['median_seconds'])
            lo, hi = paired_ci(old['samples_seconds'], new['samples_seconds'])
            row[mode+'_paired_bootstrap95_low_pct'] = lo
            row[mode+'_paired_bootstrap95_high_pct'] = hi
            row[mode+'_H100_TF32_performance_pct'] = 100*h['summary']['tf32']['median_seconds']/new['median_seconds']
        row['PPU_timing_source']=timing_sources['ppu',*key]
        row['H100_timing_source']=timing_sources['h100',*key]
        times.append(row)
        for metric in METRICS:
            c = {'input':p['case'], 'tokens':p['N_token'], 'model':p['model'], 'metric':metric,
                 'H100_native_FP32':h['summary']['fp32']['five_confidence_metrics'][metric],
                 'H100_native_TF32':h['summary']['tf32']['five_confidence_metrics'][metric],
                 'PPU_FlagGems_FP32':p['summary']['fp32']['five_confidence_metrics'][metric],
                 'PPU_FlagGems_TF32':p['summary']['tf32']['five_confidence_metrics'][metric]}
            for mode in modes:
                c['PPU_'+mode] = p['summary'][mode]['five_confidence_metrics'][metric]
                c[mode+'_delta_vs_current_PPU_TF32'] = c['PPU_'+mode]-c['PPU_FlagGems_TF32']
            for mode, prefix in [('fp32','FP32'),('tf32','TF32')]:
                expected=h['summary'][mode]['five_confidence_metrics'][metric]
                actual=p['summary'][mode]['five_confidence_metrics'][metric]
                c['PPU_vs_H100_native_'+prefix+'_absolute_delta']=actual-expected
                c['PPU_vs_H100_native_'+prefix+'_relative_difference_pct']=100*abs(actual-expected)/max(abs(expected),1e-8)
            confidence.append(c)
        gates = frozen['cross_device_gates']
        a = {'input':p['case'], 'model':p['model'],
             'PPU_FP32_vs_H100_native_FP32_status':gates['native_fp32']['status'],
             'PPU_FP32_vs_H100_native_FP32_RMSD_lower_A':gates['native_fp32']['rmsd']['lower_A'],
             'PPU_FP32_vs_H100_native_FP32_RMSD_upper_A':gates['native_fp32']['rmsd']['upper_A'],
             'PPU_TF32_vs_H100_native_TF32_status':gates['native_tf32']['status'],
             'PPU_TF32_vs_H100_native_TF32_RMSD_lower_A':gates['native_tf32']['rmsd']['lower_A'],
             'PPU_TF32_vs_H100_native_TF32_RMSD_upper_A':gates['native_tf32']['rmsd']['upper_A'],
             'candidate_vs_current_PPU_TF32_coordinate_RMSD_A':0,
             'candidate_vs_current_PPU_TF32_all_five_scores_delta':0,
             'evidence':'Inherited formal intervals after all four frozen coordinate and five-score hashes matched; raw coordinates remain on original servers.'}
        a.update(frozen['same_device_structure'])
        a.pop('tokens',None)
        accuracy.append(a)
    out = ROOT/('results_'+args.stage)
    out.mkdir(exist_ok=True)
    write_csv(out/'timings.csv', times)
    write_csv(out/'confidence.csv', confidence)
    write_csv(out/'accuracy.csv', accuracy)
    overview = {}
    for mode in sorted(candidate_modes):
        gains = [r[mode+'_time_reduction_pct'] for r in times]
        overview[mode] = {'case_count':len(gains),'positive_median_count':sum(g>0 for g in gains),
            'positive_paired_bootstrap95_count':sum(r[mode+'_paired_bootstrap95_low_pct']>0 for r in times),
            'minimum_time_reduction_pct':min(gains),'maximum_time_reduction_pct':max(gains),
            'geometric_mean_time_reduction_pct':100*(1-math.exp(statistics.mean(math.log(1-g/100) for g in gains))),
            'median_H100_native_TF32_performance_pct':statistics.median(r[mode+'_H100_TF32_performance_pct'] for r in times)}
    summary = {'status':'PASS','timings':times,'candidate_overview':overview,
               'original_predictions_and_five_confidence_exactly_retained':True,
               'full_cross_device_input_checkpoint_assets_and_ordered_rng_audit':'PASS',
               'formal_gate_counts':{m:dict(Counter(r['cross_device_gates'][m]['status'] for r in fp.values())) for m in ('native_fp32','native_tf32')},
               'protocol':'InferenceRunner.predict, FP32 dtype; TF32 permission changes matmul; uniform RNG seed 101, cycle=1/step=1/sample=1. Compile, audit instrumentation, poisoned first warmup and output export excluded from clean repeated timings.',
               'limits':['Small repeated sample count; paired bootstrap intervals describe this run only.',
                         'Formal RMSD intervals reused because frozen full predictions retained, not a fresh raw cross-device coordinate export.',
                         'No claim of full production diffusion sampling, full training acceptance, MFU or 80% target.',
                         'Rule has no input/model whitelist; tested coverage is limited to the audited model/input rows in this report.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}
    (out/'source_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('REPORT_PASS',args.stage,len(times),overview,flush=True)


if __name__ == '__main__':
    main()
