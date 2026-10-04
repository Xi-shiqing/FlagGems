"""Build complete four-mode tables from scalar-only server evidence."""
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parent
MODEL = 'protenix_base_default_v1.0.0'
METRICS = ('plddt', 'gpde', 'ptm', 'iptm', 'ranking_score')
MODES = ('h100_fp32', 'h100_tf32', 'ppu_fp32', 'ppu_tf32', 'ppu_candidate_tf32')


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT/name).open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def bootstrap(a, b):
    assert len(a) == len(b) == 10
    rng = random.Random(101)
    values = []
    for _ in range(10000):
        indices = [rng.randrange(10) for _ in range(10)]
        values.append(100*(1-statistics.median(b[i] for i in indices)/statistics.median(a[i] for i in indices)))
    values.sort()
    return {'latency_reduction_percent': 100*(1-statistics.median(b)/statistics.median(a)),
            'paired_bootstrap_95pct_ci': [values[249], values[9749]], 'draws': 10000, 'seed': 101}


def delta(a, b):
    return {key: {'absolute': a[key]-b[key],
                  'relative_percent': abs((a[key]-b[key])/b[key])*100 if b[key] else None,
                  'pass_5pct': abs(a[key]-b[key]) <= .05*abs(b[key]) if b[key] else a[key] == b[key]}
            for key in METRICS}


def main():
    p, h = read(ROOT/'ppu/derived.json'), read(ROOT/'h100/derived.json')
    assert p['checkpoint']['sha256'] == h['checkpoint']['sha256']
    old = read(ROOT.parent/'vendor_gemm_20260930/full_gems_summary.json')['rows']
    inference, confidence, structure, audits = [], [], [], {}
    for case in ('G01', 'G02', 'G03'):
        pr, hr = p['cases'][case], h['cases'][case]
        for key in ('features', 'atom_identity_sha256', 'canonical_input_sha256', 'asset_hashes'):
            assert pr['input_audit'][key] == hr['input_audit'][key], (case, key)
        for key in ('initialization_semantic_sha256', 'prediction_semantic_sha256'):
            assert pr['rng_audit'][key] == hr['rng_audit'][key], (case, key)
        summaries = {mode: (hr if mode.startswith('h100') else pr)['summary'][
            {'ppu_tf32': 'baseline_tf32', 'ppu_candidate_tf32': 'candidate_tf32'}.get(mode, mode.split('_',1)[1])]
                     for mode in MODES}
        for row in summaries.values():
            assert row['count'] == 10 and row['warmup_count'] == 3
        assert pr['environment']['use_flag_gems'] and pr['environment']['flag_gems_exclude'] == []
        assert not hr['environment']['use_flag_gems']
        comparison = bootstrap(summaries['ppu_tf32']['samples_seconds'], summaries['ppu_candidate_tf32']['samples_seconds'])
        meta = {'input': case, 'model': MODEL, 'N_token': pr['input_audit']['N_token']}
        times = {mode+'_seconds': summaries[mode]['median_seconds'] for mode in MODES}
        inference.append(meta | times | {'candidate_latency_reduction_percent': comparison['latency_reduction_percent'],
            'candidate_gain_ci95_low_percent': comparison['paired_bootstrap_95pct_ci'][0],
            'candidate_gain_ci95_high_percent': comparison['paired_bootstrap_95pct_ci'][1],
            'ppu_tf32_over_h100_tf32_time': times['ppu_tf32_seconds']/times['h100_tf32_seconds'],
            'ppu_candidate_tf32_over_h100_tf32_time': times['ppu_candidate_tf32_seconds']/times['h100_tf32_seconds'],
            'warmups_per_configuration': 3, 'repeats_per_configuration': 10,
            'scope': 'InferenceRunner.predict; diagnostic cycle=step=sample=1; initialization/preprocessing/output saving excluded'})
        for mode in MODES:
            confidence.append(meta | {'mode': mode} | summaries[mode]['five_confidence_metrics'] |
                              {'prediction_sha256': summaries[mode]['prediction_digest']['sha256']})
        frozen = {mode: next(r for r in old if r['case'] == case and r['model'] == MODEL and r['mode'] == mode)
                  for mode in MODES[:-1]}
        for mode in MODES[:-1]:
            record = summaries[mode]
            # Inherit an exact distance only when the prediction bytes match.
            if mode != 'h100_fp32':
                assert record['prediction_digest']['sha256'] == frozen[mode]['prediction_sha256'], (case, mode)
        for i, mode in enumerate(MODES):
            for ref in MODES[:i]:
                same = (mode == 'ppu_fp32' and ref == 'h100_fp32') or (mode.endswith('tf32') and ref == 'h100_tf32')
                d = delta(summaries[mode]['five_confidence_metrics'], summaries[ref]['five_confidence_metrics'])
                exact = None
                source = 'Direct same-server measurement'
                if mode.startswith('ppu') and ref.startswith('ppu'):
                    a, b = {'ppu_fp32': 'fp32', 'ppu_tf32': 'baseline_tf32', 'ppu_candidate_tf32': 'candidate_tf32'}[mode], {'ppu_fp32': 'fp32', 'ppu_tf32': 'baseline_tf32'}[ref]
                    exact = pr['same_device_rmsd_A'].get(a+'_vs_'+b, pr['same_device_rmsd_A'].get(b+'_vs_'+a))
                elif mode == 'h100_tf32' and ref == 'h100_fp32':
                    exact = hr['same_device_rmsd_A']['fp32_vs_tf32']
                else:
                    base_mode = 'ppu_tf32' if mode == 'ppu_candidate_tf32' else mode
                    distance = frozen[base_mode]['vs_'+ref]['structure']['kabsch_rmsd_A']
                    changes = []
                    if mode == 'ppu_candidate_tf32':
                        changes.append(pr['same_device_rmsd_A']['baseline_tf32_vs_candidate_tf32'])
                    if ref == 'h100_fp32':
                        changes.append(hr['drift_vs_frozen']['fp32']['rmsd_vs_frozen_same_precision_A'])
                    if not changes:
                        exact, source = distance, 'Exact frozen distance; both current prediction hashes identical to corresponding frozen outputs'
                    else:
                        low, high = max(0, distance-sum(changes)), distance+sum(changes)
                        source = 'Rigorous triangle-inequality interval from exact frozen distance and directly measured same-server drifts; not direct cross-server candidate RMSD'
                if exact is not None:
                    low = high = exact
                row = meta | {'mode': mode, 'reference': ref,
                    'purpose': 'same_precision_primary' if same else 'diagnostic',
                    'rmsd_exact_A': exact, 'rmsd_lower_bound_A': low, 'rmsd_upper_bound_A': high,
                    'rmsd_method': source,
                    'rmsd_gate_0_1A': 'PASS' if high <= .1 else 'FAIL' if low > .1 else 'UNRESOLVED',
                    'five_confidence_gate_5pct': all(v['pass_5pct'] for v in d.values())}
                for key, value in d.items():
                    row[key+'_absolute_delta'] = value['absolute']
                    row[key+'_relative_percent'] = value['relative_percent']
                structure.append(row)
        audits[case] = {'inputs_and_rng_cross_device': 'PASS', 'ppu': pr['input_audit'],
            'rng_ppu': pr['rng_audit'], 'rng_h100': hr['rng_audit'],
            'performance_comparison': comparison,
            'fp32_prediction_unchanged': summaries['ppu_fp32']['prediction_equal_reference'],
            'native_tf32_prediction_unchanged': summaries['h100_tf32']['prediction_equal_reference']}
    write_csv('four_mode_inference.csv', inference)
    write_csv('four_mode_confidence.csv', confidence)
    write_csv('four_mode_structure.csv', structure)
    wide = []
    for case in ('G01', 'G02', 'G03'):
        values = {r['mode']: r for r in confidence if r['input'] == case}
        for key in METRICS:
            row = {'input': case, 'model': MODEL, 'metric': key} | {m: values[m][key] for m in MODES}
            for mode, ref in [('ppu_fp32','h100_fp32'), ('ppu_tf32','h100_tf32'), ('ppu_candidate_tf32','h100_tf32')]:
                difference = values[mode][key]-values[ref][key]
                row[mode+'_minus_'+ref] = difference
                row[mode+'_relative_percent'] = abs(difference/values[ref][key])*100 if values[ref][key] else None
            wide.append(row)
    write_csv('four_mode_confidence_wide.csv', wide)
    training = list(csv.DictReader((ROOT.parent/'jittor_reference_20261001/four_mode_training.csv').open()))
    write_csv('four_mode_training_historical.csv', training)
    report = {'inference': inference, 'confidence': confidence, 'structure': structure,
        'input_rng_checkpoint_audit': audits, 'checkpoint': p['checkpoint'],
        'runtime_precision_map': {s: read(ROOT/s/'derived.json')['cases'] for s in ('ppu','h100')},
        'training': {'scope': 'Historical four standalone operator probes; not fresh timing or full-model training', 'rows': training},
        'missing': ['Nine-model candidate acceptance', 'Full default inference protocol',
                    'Full-model training gradients/updates', 'MFU',
                    'Direct cross-server RMSD for numerically changed candidate (rigorous bounds reported instead)',
                    'First cross-platform TF32 activation divergence and causal repair'],
        'candidate_policy': 'Experimental unsplit/stages3 ordinary TF32; FP32 unchanged; no production-default adoption',
        'historical_fp32_correction': 'Old native H100 FP32 disabled matmul TF32 but kept cuDNN TF32 enabled; this batch disables both. G02 single-factor intervention reproduces both hashes.'}
    report['flag_gems_scope'] = 'PPU FlagGems context enabled with exclude=[]; verified MM/addmm/BMM and public Triangle Attention entries. This is not 100% model-operator replacement.'
    if (ROOT/'ppu/sdpa_registry.json').exists():
        report['sdpa_registry_audit'] = read(ROOT/'ppu/sdpa_registry.json')
        report['native_sdpa_coverage_gap'] = 'Installed FlagGems only registers _flash_attention_forward. G02 performs 24 native efficient-SDPA calls and 67 math-SDPA calls per prediction; math dispatch may use registered FlagGems primitives.'
    if (ROOT/'profile_attribution.json').exists():
        report['profile_attribution'] = read(ROOT/'profile_attribution.json')
    report['training']['triangle_attention_native_reference'] = 'Historical H100 training probe uses native Torch QK-softmax-PV, not the cuEquivariance inference backend. Full-model training remains unmeasured.'
    if (ROOT/'ppu/shadow/G02/shadow.json').exists():
        report['same_activation_shadow'] = read(ROOT/'ppu/shadow/G02/shadow.json')
    if (ROOT/'ppu/shadow/G02/integrity.json').exists():
        report['shadow_teacher_integrity'] = read(ROOT/'ppu/shadow/G02/integrity.json')
    if (ROOT/'h100/controls.json').exists():
        report['native_cudnn_controls'] = read(ROOT/'h100/controls.json')
    (ROOT/'complete_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ('.py','.sh','.json','.csv') and p.name != 'manifest.json'}
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    assert len(inference) == 3 and len(confidence) == 15 and len(structure) == 30
    print('COMPLETE_FOUR_MODE_REPORT_OK', len(inference), len(confidence), len(structure))
    for row in inference:
        print('TIMING', row)
    for row in structure:
        if row['purpose'] == 'same_precision_primary':
            print('QUALITY', row['input'], row['mode'], row['rmsd_exact_A'], row['rmsd_lower_bound_A'], row['rmsd_upper_bound_A'], row['rmsd_gate_0_1A'])


if __name__ == '__main__':
    main()
