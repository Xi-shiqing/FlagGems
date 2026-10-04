"""Join scalar/hash evidence into complete four-mode diagnostic tables."""
import csv
import json
from pathlib import Path
import random
import statistics


WORK = Path(__file__).resolve().parent
PRIOR = WORK.parent/'matched_tf32_20261001'
METRICS = ('plddt', 'gpde', 'ptm', 'iptm', 'ranking_score')


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    with (WORK/name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_savings(baseline, candidate):
    assert len(baseline) == len(candidate) == 10
    rng = random.Random(101)
    samples = []
    for _ in range(10000):
        ids = [rng.randrange(10) for _ in range(10)]
        samples.append(100*(1-statistics.median(candidate[i] for i in ids)/
                            statistics.median(baseline[i] for i in ids)))
    samples.sort()
    return {'observed_time_reduction_percent': 100*(1-statistics.median(candidate)/statistics.median(baseline)),
            'paired_bootstrap_95_percent_interval': [samples[249], samples[9749]],
            'scope': '10 paired measured repetitions in this batch; 10000 resamples, seed101'}


def main():
    ppu = read(WORK/'ppu/interventions_derived.json')
    h100 = read(WORK/'h100/routes_derived.json')
    ppu_routes = read(WORK/'ppu/routes/G02/routes.json')
    h100_routes = read(WORK/'h100/routes/G02/routes.json')
    ppu_control = read(WORK/'ppu/interventions/G02/routes.json')
    assert ppu['input_audit'] == h100['input_audit']
    for key in ('initialization_semantic_sha256', 'prediction_semantic_sha256',
                'initialization_events', 'events_per_prediction'):
        assert ppu['rng_audit'][key] == h100['rng_audit'][key]
    assert ppu['checkpoint_sha256'] == h100['checkpoint_sha256']
    # Use only the old exact native-TF32 distance whose predictions remain hash-identical.
    prior_structure = list(csv.DictReader((PRIOR/'four_mode_structure.csv').open()))
    native = next(x for x in prior_structure if x['input'] == 'G02'
                  and x['mode'] == 'ppu_tf32' and x['reference'] == 'h100_tf32')
    fp32_bound = next(x for x in prior_structure if x['input'] == 'G02'
                     and x['mode'] == 'ppu_fp32' and x['reference'] == 'h100_fp32')
    assert native['rmsd_method'].startswith('Exact frozen distance')
    baseline_distance = float(native['rmsd_exact_A'])
    strict_fp32_upper = float(fp32_bound['rmsd_upper_bound_A'])
    native_tf32_vs_fp32 = h100['rmsd_vs_local_baseline_A']['tf32']['fp32']
    summaries = {'H100_native_fp32': h100['summary']['fp32'],
                 'H100_native_tf32': h100['summary']['tf32'],
                 'PPU_FlagGems_fp32': ppu['summary']['fp32'],
                 'PPU_FlagGems_tf32': ppu['summary']['tf32'],
                 **{'PPU_TF32_'+k: v for k, v in ppu['summary'].items() if k not in ('fp32', 'tf32')}}
    modes, confidence = [], []
    for mode, row in summaries.items():
        ref_mode = 'fp32' if mode.endswith('_fp32') else 'tf32'
        ref = h100['summary'][ref_mode]
        item = {'case': 'G02', 'N_token': 245, 'mode': mode,
                'median_seconds': row['median_seconds'], 'warmup_count': row['warmup_count'],
                'repeat_count': row['count'], 'reference': 'H100_native_'+ref_mode,
                'time_ratio_to_matching_H100': row['median_seconds']/ref['median_seconds'],
                **row['five_confidence_metrics']}
        errors = {}
        for key in METRICS:
            value, reference = row['five_confidence_metrics'][key], ref['five_confidence_metrics'][key]
            absolute = value-reference
            relative = abs(absolute)/abs(reference)*100 if reference else None
            passed = relative <= 5 if relative is not None else absolute == 0
            confidence.append({'case': 'G02', 'mode': mode, 'metric': key,
                'value': value, 'reference': 'H100_native_'+ref_mode,
                'reference_value': reference, 'signed_absolute_difference': absolute,
                'relative_error_percent': relative, 'pass_5_percent': passed})
            errors[key] = relative
        item['maximum_confidence_relative_error_percent'] = max(x for x in errors.values() if x is not None)
        if mode.startswith('H100'):
            item.update({'rmsd_lower_A': 0., 'rmsd_upper_A': 0., 'rmsd_kind': 'self', 'structure_gate': 'REFERENCE'})
        elif ref_mode == 'fp32':
            item.update({'rmsd_lower_A': 0., 'rmsd_upper_A': float(fp32_bound['rmsd_upper_bound_A']),
                         'rmsd_kind': 'inherited rigorous bound; both prediction hashes unchanged', 'structure_gate': 'PASS'})
        else:
            variant = mode.removeprefix('PPU_TF32_') if mode.startswith('PPU_TF32_') else 'tf32'
            drift = ppu['rmsd_vs_local_baseline_A'][variant]['tf32']
            lower, upper = max(0., baseline_distance-drift), baseline_distance+drift
            fp_drift = ppu['rmsd_vs_local_baseline_A'][variant]['fp32']
            lower = max(lower, abs(native_tf32_vs_fp32-fp_drift)-strict_fp32_upper)
            upper = min(upper, native_tf32_vs_fp32+fp_drift+strict_fp32_upper)
            if variant == 'tf32':
                lower = upper = baseline_distance
            item.update({'rmsd_lower_A': lower, 'rmsd_upper_A': upper,
                'rmsd_kind': 'inherited exact distance' if variant == 'tf32' else 'intersection of rigorous triangle-inequality intervals using FP32 and TF32 anchors',
                'structure_gate': 'PASS' if upper <= .1 else ('FAIL' if lower > .1 else 'UNRESOLVED')})
        modes.append(item)
    operators, precision_map = [], []
    synthetic = {server: read(WORK/server/'synthetic.json') for server in ('ppu', 'h100')}
    assert all(x['status'] == 'PASS' and len(x['records']) == 4 for x in synthetic.values())
    for pp, hh in zip(synthetic['ppu']['records'], synthetic['h100']['records']):
        assert pp['name'] == hh['name']
        assert pp['inputs'] == hh['inputs'], pp['name']
        row = {'entry': pp['name'], 'op': pp['op'],
               'input_shapes': json.dumps([x['shape'] for x in pp['inputs']]),
               'input_hashes_match': True,
               'IEEE_FP64_gold_hashes_match': pp['gold_ieee_fp64_sha256'] == hh['gold_ieee_fp64_sha256']}
        for server, record in (('H100', hh), ('PPU', pp)):
            for precision in ('fp32', 'tf32'):
                prefix = server+'_'+precision
                data = record['modes'][precision]
                row[prefix+'_median_ms'] = data['median_ms']
                row[prefix+'_relative_l2_vs_ieee_fp64'] = data['vs_ieee_fp64']['relative_l2']
                row[prefix+'_relative_l2_vs_rne_tf32_fp64'] = data['vs_rne_tf32_fp64']['relative_l2']
        operators.append(row)
    for pp in ppu_routes['routes']['tf32']['replays']:
        hh = next((x for x in h100_routes['routes']['tf32']['replays']
                   if x['op'] == pp['op'] and [t['shape'] for t in x['inputs']] == [t['shape'] for t in pp['inputs']]), None)
        if hh:
            precision_map.append({'op': pp['op'], 'input_shapes': [t['shape'] for t in pp['inputs']],
                'real_input_hashes_match': all(a['sha256'] == b['sha256'] for a, b in zip(pp['inputs'], hh['inputs'])),
                'PPU_local_tf32_vs_fp32_relative_l2': pp['tf32_vs_fp32']['relative_l2'],
                'H100_local_tf32_vs_fp32_relative_l2': hh['tf32_vs_fp32']['relative_l2'],
                'PPU_stack': pp['stack'], 'H100_stack': hh['stack']})
    early = {server: read(WORK/server/'early_linear/G02/routes.json') for server in ('ppu', 'h100')}
    integrity = {server: read(WORK/server/'early_linear/G02/integrity.json') for server in ('ppu', 'h100')}
    assert all(x['status'] == 'PASS' for x in integrity.values())
    early_rows = []
    for pp, hh in zip(early['ppu']['linear'], early['h100']['linear']):
        assert pp['module'] == hh['module']
        row = {'module': pp['module'], 'input_shapes': json.dumps([x['shape'] for x in pp['inputs']]),
               'real_input_weight_hashes_match': all(a['sha256'] == b['sha256'] for a,b in zip(pp['inputs'],hh['inputs']))}
        for server, record in (('H100', hh), ('PPU', pp)):
            for precision in ('fp32', 'tf32'):
                data = record['modes'][precision]
                row[server+'_'+precision+'_median_ms'] = data['median_ms']
                row[server+'_'+precision+'_sample_relative_l2_vs_ieee_fp64'] = data['sample_vs_ieee_fp64']['relative_l2']
                row[server+'_'+precision+'_relative_l2_vs_local_fp32'] = 0. if precision == 'fp32' else record['tf32_vs_fp32']['relative_l2']
            row[server+'_local_tf32_vs_fp32_relative_l2'] = record['tf32_vs_fp32']['relative_l2']
        early_rows.append(row)
    write_csv('four_mode_model.csv', modes)
    write_csv('five_confidence_differences.csv', confidence)
    write_csv('four_mode_operators.csv', operators)
    write_csv('four_mode_early_linear.csv', early_rows)
    report = {'status': 'COMPLETE_DIAGNOSTIC_ROUND', 'model': 'protenix_base_default_v1.0.0',
        'input': 'G02; N_token=245; N_atom=2529', 'protocol': 'float32 storage; seed101; cycle/step/sample=1; MSA on; templates/RNA MSA off; 3 warmups+10 uninstrumented repeats per model policy',
        'same_precision_gate': 'PPU TF32 vs unchanged H100 native TF32: all five confidence <=5%; RMSD<=0.1 A. FP32 compared independently.',
        'model_comparison': modes, 'confidence': confidence, 'synthetic_operator_comparison': operators,
        'structure_bound_anchor_evidence': {'native_tf32_vs_strict_fp32_exact_A': native_tf32_vs_fp32,
            'ppu_fp32_vs_native_fp32_upper_A': strict_fp32_upper,
            'ppu_ordinary_tf32_vs_native_tf32_inherited_exact_A': baseline_distance,
            'source': str(PRIOR/'four_mode_structure.csv'), 'inherited_predictions_current_hashes_unchanged': True},
        'intervention_time_effects': {mode: paired_savings(ppu['summary']['tf32']['samples_seconds'], value['samples_seconds'])
                                    for mode, value in ppu['summary'].items() if mode not in ('fp32', 'tf32')},
        'real_entry_precision_map': precision_map, 'early_linear_comparison': early_rows,
        'input_checkpoint_rng_audits': {'ppu': ppu, 'h100': h100},
        'raw_scalar_evidence': {'ppu_routes': ppu_routes, 'h100_routes': h100_routes,
                                'ppu_interventions': ppu_control, 'synthetic': synthetic, 'early_linear': early,
                                'early_linear_integrity': integrity},
        'limitations': ['Synthetic operands use real shapes/strides but are not real activations.',
            'Local BMM replays use identical local operands; most real activations differ across platforms.',
            'New cross-server intervention RMSD is bounded rather than directly measured.',
            'Native Efficient SDPA remains native PPU Torch; it is not a FlagGems implementation.',
            'No new full-model training, nine-model acceptance, full-default inference, or MFU measurement.',
            'No production default changed; operator-family IEEE interventions are diagnostics.'],
        'prior_round': str(PRIOR/'complete_comparison.json')}
    (WORK/'complete_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print('REPORT_PASS', len(modes), len(confidence), len(operators), len(early_rows))


if __name__ == '__main__':
    main()
