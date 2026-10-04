"""Join scalar/hash evidence; keep every precision and reference explicit."""
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics


ROOT = Path(__file__).resolve().parent
METRICS = ('plddt', 'gpde', 'ptm', 'iptm', 'ranking_score')
MODES = ('h100_fp32', 'h100_tf32', 'ppu_fp32', 'ppu_tf32', 'ppu_candidate_tf32')


def read(path):
    return json.loads(path.read_text())


def csv_write(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT/name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gain(before, after):
    assert len(before) == len(after)
    rng = random.Random(101)
    values = []
    for _ in range(10000):
        index = [rng.randrange(len(before)) for _ in before]
        values.append((1-statistics.median(after[i] for i in index)/
                       statistics.median(before[i] for i in index))*100)
    values.sort()
    return {'latency_reduction_percent': (1-statistics.median(after)/statistics.median(before))*100,
            'paired_bootstrap_ci95_percent': [values[249], values[9749]]}


def main():
    inventory = read(ROOT/'ppu/source_inventory.json')
    assert inventory['candidate/rank_one'] == hashlib.sha256((ROOT/'operator_measured_rank_one.py').read_bytes()).hexdigest()
    current = read(ROOT/'ppu/validation_source_inventory.json')
    assert current['rank_one_sha256'] == hashlib.sha256((ROOT/'rank_one.py').read_bytes()).hexdigest()
    assert current['final_check_exit'] == 0
    assert read(ROOT/'ppu/check.json')['status'] == 'PASS'
    for name, values in read(ROOT/'ppu/candidate_sources.json').items():
        assert inventory['candidate/'+name] == values['after']
        assert inventory['baseline/'+name] == values['before']
    p, h = [read(ROOT/server/'operators.json') for server in ('ppu', 'h100')]
    assert p['status'] == h['status'] == 'PASS'
    zero_checks = {s:read(ROOT/s/'signed_zero.json') for s in ('ppu', 'h100')}
    for pp, hh in zip(zero_checks['ppu']['cases'], zero_checks['h100']['cases']):
        assert pp['name'] == hh['name'] and pp['canonical_zero_product_sha256'] == hh['canonical_zero_product_sha256']
    operators, errors, operator_gains = [], [], []
    for pp, hh in zip(p['cases'], h['cases']):
        assert pp['name'] == hh['name'] and pp['shape'] == hh['shape']
        assert pp['input_sha256'] == hh['input_sha256'] and pp['gold_sha256'] == hh['gold_sha256']
        assert pp['candidate_fp32_bitwise_equal_baseline']
        values = {**{'h100_'+k:v for k, v in hh['modes'].items()},
                  **{'ppu_'+k:v for k, v in pp['modes'].items()}}
        row = {'operator': pp['name'], 'kind': pp['kind'], 'shape': str(pp['shape']),
               'input_sha256_equal': True, 'candidate_fp32_bitwise_equal': True}
        for mode in MODES:
            for kind in ('inference', 'training'):
                row[mode+'_'+kind+'_ms'] = values[mode]['median_'+kind+'_ms']
            for field, error in values[mode]['errors_vs_cpu_fp64'].items():
                errors.append({'operator': pp['name'], 'shape': str(pp['shape']), 'mode': mode,
                               'field': field, **error,
                               'reference': 'CPU FP64 full-tensor gold; updated parameters rounded once to FP32',
                               'same_hash_as_h100_tf32': error['sha256'] == values['h100_tf32']['errors_vs_cpu_fp64'][field]['sha256']})
        for kind in ('inference', 'training'):
            operator_gains.append({'operator': pp['name'], 'kind': kind,
                **gain(pp['modes']['tf32']['samples_ms'][kind+'_ms'],
                       pp['modes']['candidate_tf32']['samples_ms'][kind+'_ms']),
                'policy_changes_this_case': pp['name'] != 'mm_k3_regression'})
        operators.append(row)

    p, h = [read(ROOT/server/'derived.json') for server in ('ppu', 'h100')]
    old = read(ROOT/'previous_matched_anchors.json')
    assert p['checkpoint']['sha256'] == h['checkpoint']['sha256']
    inference, confidence, structure, audits = [], [], [], {}
    for case in ('G01', 'G02', 'G03'):
        pp, hh = p['cases'][case], h['cases'][case]
        assert pp['input_audit'] == hh['input_audit'], case
        for key in ('initialization_semantic_sha256', 'prediction_semantic_sha256'):
            assert pp['rng_audit'][key] == hh['rng_audit'][key], (case, key)
        for mode in ('fp32', 'tf32'):
            assert hh['summary'][mode]['prediction_digest'] == old['prediction_digests']['h100'][case][mode]
        for mode, prior in (('fp32', 'fp32'), ('baseline_tf32', 'baseline_tf32')):
            assert pp['summary'][mode]['prediction_digest'] == old['prediction_digests']['ppu'][case][prior]
        values = {**{'h100_'+k:v for k, v in hh['summary'].items()},
                  **{('ppu_tf32' if k == 'baseline_tf32' else 'ppu_'+k):v for k, v in pp['summary'].items()}}
        row = {'input': case, 'model': 'protenix_base_default_v1.0.0', 'N_token': pp['input_audit']['N_token'],
               'protocol': 'InferenceRunner.predict; cycle=step=sample=1; alternating precisions; initializations excluded'}
        for mode in MODES:
            row[mode+'_seconds'] = values[mode]['median_seconds']
            row[mode+'_warmups'] = values[mode]['warmup_count']
            row[mode+'_repeats'] = values[mode]['count']
            confidence.append({'input': case, 'mode': mode, **values[mode]['five_confidence_metrics'],
                               'prediction_sha256': values[mode]['prediction_digest']['sha256']})
        row.update(gain(values['ppu_tf32']['samples_seconds'], values['ppu_candidate_tf32']['samples_seconds']))
        row['ppu_candidate_tf32_over_h100_tf32_time'] = row['ppu_candidate_tf32_seconds']/row['h100_tf32_seconds']
        row['validation_scope'] = '10 repeats' if case == 'G02' else '3-repeat exploratory multi-input check; not final acceptance'
        inference.append(row)

        def historical(mode, reference):
            return next(x for x in old['structure'] if x['input'] == case and x['mode'] == mode and x['reference'] == reference)

        for mode, reference in (('h100_tf32', 'h100_fp32'), ('ppu_fp32', 'h100_fp32'),
                                ('ppu_tf32', 'h100_tf32'), ('ppu_candidate_tf32', 'h100_tf32'),
                                ('ppu_candidate_tf32', 'h100_fp32'), ('ppu_candidate_tf32', 'ppu_tf32'),
                                ('ppu_candidate_tf32', 'ppu_fp32')):
            exact = None
            if mode == 'h100_tf32':
                exact = hh['same_device_rmsd_A']['fp32_vs_tf32']
                lo = hi = exact
                method = 'Direct same-server Kabsch all-atom RMSD'
            elif mode == 'ppu_fp32':
                anchor = historical(mode, reference)
                lo, hi, exact = (anchor['rmsd_lower_bound_A'], anchor['rmsd_upper_bound_A'], anchor['rmsd_exact_A'])
                method = 'Reused rigorous interval; current prediction hashes match previous matched strict FP32 outputs'
            elif mode == 'ppu_tf32':
                anchor = historical(mode, reference)
                exact = anchor['rmsd_exact_A']
                assert exact is not None
                lo = hi = exact
                method = 'Reused exact cross-server distance; both current prediction hashes identical to anchor outputs'
            elif reference == 'h100_tf32':
                anchor = historical('ppu_tf32', reference)['rmsd_exact_A']
                drift = pp['same_device_rmsd_A']['baseline_tf32_vs_candidate_tf32']
                lo, hi = max(0, anchor-drift), anchor+drift
                # A second, strict-FP32 anchor can tighten the interval without
                # changing the primary H100 TF32 comparison or exporting tensors.
                fp_upper = historical('ppu_fp32', 'h100_fp32')['rmsd_upper_bound_A']
                fp_drift = pp['same_device_rmsd_A']['candidate_tf32_vs_fp32']
                native_drift = hh['same_device_rmsd_A']['fp32_vs_tf32']
                fp_lo, fp_hi = max(0, fp_drift-fp_upper), fp_drift+fp_upper
                lo = max(lo, native_drift-fp_hi, fp_lo-native_drift)
                hi = min(hi, native_drift+fp_hi)
                method = 'Intersection of rigorous triangle-inequality intervals from unchanged TF32/strict-FP32 anchors and same-server drifts; not a direct candidate cross-server measurement'
            elif reference == 'h100_fp32':
                drift = pp['same_device_rmsd_A']['candidate_tf32_vs_fp32']
                anchor = historical('ppu_fp32', reference)
                lo, hi = max(0, drift-anchor['rmsd_upper_bound_A']), drift+anchor['rmsd_upper_bound_A']
                method = 'Rigorous triangle-inequality interval using unchanged strict FP32 anchor'
            else:
                key = 'baseline_tf32_vs_candidate_tf32' if reference == 'ppu_tf32' else 'candidate_tf32_vs_fp32'
                exact = pp['same_device_rmsd_A'][key]
                lo = hi = exact
                method = 'Direct same-server Kabsch all-atom RMSD'
            distance = {'input': case, 'mode': mode, 'reference': reference,
                'purpose': 'same_precision_primary' if (reference == 'h100_tf32' or mode == 'ppu_fp32') else 'diagnostic',
                'rmsd_exact_A': exact, 'rmsd_lower_bound_A': lo, 'rmsd_upper_bound_A': hi,
                'rmsd_method': method, 'rmsd_gate_0_1A': 'PASS' if hi <= .1 else 'FAIL' if lo > .1 else 'UNRESOLVED'}
            gates = []
            for metric in METRICS:
                a, b = (values[k]['five_confidence_metrics'][metric] for k in (mode, reference))
                delta = a-b
                percent = abs(delta/b)*100 if b else None
                distance[metric+'_absolute_delta'] = delta
                distance[metric+'_relative_percent'] = percent
                gates.append(percent <= 5 if percent is not None else delta == 0)
            distance['five_confidence_gate_5pct'] = all(gates)
            structure.append(distance)
        audits[case] = {'input': pp['input_audit'], 'ppu_rng': pp['rng_audit'], 'h100_rng': hh['rng_audit'],
                        'baseline_hashes_unchanged_vs_previous_matched_round': True,
                        'rank_one_calls_first_candidate_warmup': read(ROOT/'ppu/model'/case/'paired.json')['rank_one_calls_first_candidate_warmup']}

    report = {'operator_timings': operators, 'operator_errors': errors, 'operator_gains': operator_gains,
              'inference': inference, 'confidence': confidence, 'structure': structure,
              'input_rng_checkpoint_audit': audits, 'checkpoint': p['checkpoint'],
              'candidate_policy': 'PPU float32 storage under allow_tf32=True; reduction K=1 uses IEEE vector outer product in MM/addmm/BMM and out APIs. No exact model/input/shape table. FP32 and K>1 paths retained.',
              'training_scope': 'Standalone forward/backward/SGD operator checks only; generated Torch backward still uses original MM/BMM policy for K>1. Full model training and nine-model acceptance not run.',
              'deployment': 'Isolated frozen-server candidate; source integration in operator repository. Production server package unchanged. No commit/push in this round.',
              'source_validation': 'Exact shared helper GPU-tested on installed frozen package and all six real Torch entries. Current newer operator-repository integration passed syntax and AST entry checks; the entire newer package was not GPU-loaded.',
              'structure_scope': 'All-atom Kabsch RMSD with matching atom-order hashes; candidate cross-server RMSD reported as intervals, not direct measurements. Raw tensors remain on their original servers.',
              'runtime_precision': {'ppu': {k:v['runtime_precision'] for k,v in p['cases'].items()},
                                    'h100': {k:v['runtime_precision'] for k,v in h['cases'].items()}},
              'operator_signed_zero_audit': zero_checks}
    report['model_measurement_sources'] = read(ROOT/'ppu/model_measurement_sources.json')
    report['post_model_index_fix'] = 'Promote program-id to int64 before multiplication and include bias strides in range detection. All measured shapes use int32 indexing; normal dense/strided entries and both int64 bias/no-bias branches rechecked. Exact measured model/operator helper sources archived separately.'
    report['operator_repeatability_before_index_fix'] = read(ROOT/'ppu/operators_before_index_fix.json')
    report['performance_limit'] = 'Small BMM timing benefit is not stable across two measurement rounds. Do not claim a general BMM speedup or an end-to-end model speedup.'
    (ROOT/'complete_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    csv_write('four_mode_operator_timings.csv', operators)
    csv_write('four_mode_operator_errors.csv', errors)
    csv_write('operator_gains.csv', operator_gains)
    csv_write('four_mode_inference.csv', inference)
    csv_write('four_mode_confidence.csv', confidence)
    csv_write('four_mode_structure.csv', structure)
    protected = ROOT.parent.parent/'ppu_mean_fix_20260920/benchmark_e2e.py'
    protected_hash = hashlib.sha256(protected.read_bytes()).hexdigest() if protected.exists() else None
    if protected_hash is not None:
        assert protected_hash == '6ede90f98d9435b8509fb37b086c6a490296b49a99c6fbbd4094eda2f5506d42'
    manifest = {'status': 'VALIDATED_CANDIDATE_NOT_FULL_TF32_ACCEPTANCE',
                'sources_sha256': {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in ROOT.iterdir() if x.suffix in ('.py', '.sh')},
                'scope': report['candidate_policy'], 'evidence': 'Derived JSON/hash scalars only',
                'protected_benchmark_sha256': protected_hash,
                'server_source_inventory': inventory,
                'final_validation_source_inventory': current}
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('RANK_ONE_REPORT_PASS', len(operators), len(inference), len(confidence), len(structure))


if __name__ == '__main__':
    main()
