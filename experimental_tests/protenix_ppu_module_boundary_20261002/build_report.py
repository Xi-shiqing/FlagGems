"""Join scalar/hash evidence; coordinate sketches never determine acceptance."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS = ('plddt', 'gpde', 'ptm', 'iptm', 'ranking_score')


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def boundary_rows(filename):
    data = {s: read(ROOT / s / filename) for s in ('ppu', 'h100')}
    flat = {}
    metadata = {}
    for server, evidence in data.items():
        for mode, rows in evidence['modes'].items():
            flat[server, mode] = {}
            for row in rows:
                for side in ('inputs', 'outputs'):
                    for path, value in row[side].items():
                        key = row['module'], row['call'], side, path
                        record = dict(evidence['pool'][value['pool']])
                        record['vs_own_fp32'] = value.get('vs_own_fp32')
                        flat[server, mode][key] = record
                        metadata[key] = {k: row[k] for k in ('event', 'class', 'kind')}
    common = set.intersection(*(set(v) for v in flat.values()))
    result = []
    for key in sorted(common, key=lambda k: (metadata[k]['event'], k)):
        values = {s + '_' + m: flat[s, m][key] for s, m in flat}
        row = dict(zip(('module', 'call', 'side', 'tensor'), key)) | metadata[key]
        row['shape'] = str(values['h100_tf32']['shape'])
        for mode, value in values.items():
            row[mode + '_rms'] = value['rms']
            row[mode + '_sha256'] = value['sha256']
            if value['vs_own_fp32']:
                row[mode + '_vs_own_fp32_relative_l2_exact'] = value['vs_own_fp32'].get('relative_l2')
        for mode in ('fp32', 'tf32'):
            p, h = values['ppu_' + mode], values['h100_' + mode]
            equal = p['shape'] == h['shape'] and p['sha256'] == h['sha256']
            row[mode + '_cross_device_hash_equal'] = equal
            estimate = None
            if p['shape'] == h['shape'] and p['aggregate_checksums'] and h['aggregate_checksums']:
                estimate = math.sqrt(sum((p['aggregate_checksums'][k] - h['aggregate_checksums'][k])**2
                                         for k in p['aggregate_checksums']) / max(h['squared_norm_fp64'], 1e-60))
            row[mode + '_cross_device_relative_l2_estimate'] = estimate
            row[mode + '_cross_device_rms_error_estimate'] = None if estimate is None else estimate * h['rms']
        result.append(row)
    return result


def distance(case, a, b):
    values = case['same_device_rmsd_A']
    return values.get(a + '_vs_' + b, values.get(b + '_vs_' + a))


def triangle_rows():
    results = triangle_records()
    indexed = {s: {row['family']: row for row in rows} for s, rows in results.items()}
    rows = []
    for family in sorted(indexed['ppu'].keys() & indexed['h100'].keys()):
        probes = set.union(*(set(indexed[s][family]['replays']) for s in indexed))
        for probe in sorted(probes):
            row = {'family': family, 'probe': probe,
                   'scope': 'Each server replays its own complete actual operands; no matched-input cross-device error claim'}
            for server in indexed:
                item = indexed[server][family]
                row[server + '_input_shapes'] = str([v['shape'] for v in item['inputs']])
                row[server + '_fp32_vs_own_fp32_relative_l2'] = 0.0
                row[server + '_tf32_vs_own_fp32_relative_l2'] = item['replays']['tf32']['vs_unmodified_fp32']['relative_l2']
                if probe not in item['replays']:
                    continue
                value = item['replays'][probe]
                row[server + '_probe_output_sha256'] = value['output_sha256']
                row[server + '_probe_hash_equal_tf32'] = value['output_hash_equal_unmodified_tf32']
                for reference in ('tf32', 'fp32'):
                    for metric in ('relative_l2', 'max_abs', 'rms'):
                        row[server + '_probe_vs_' + reference + '_' + metric] = value['vs_unmodified_' + reference][metric]
            rows.append(row)
    return rows


def auxiliary_rows():
    results = triangle_records()
    indexed = {s: {row['family']: row for row in rows} for s, rows in results.items()}
    rows = []
    for family in sorted(indexed['ppu'].keys() & indexed['h100'].keys()):
        row = {'family': family, 'scope': 'Full same-input local TF32/IEEE comparison; auxiliary cache units not compared across devices'}
        for server, families in indexed.items():
            value = families[family].get('auxiliary')
            if not value:
                continue
            for mode, errors in value['output_comparison_to_normal_entry'].items():
                assert errors['max_abs'] == 0.0, (server, family, mode)
            row[server + '_fp32_lse_vs_own_fp32_relative_l2'] = 0.0
            for name, metrics in value['tf32_vs_fp32'].items():
                for metric, number in metrics.items():
                    row[server + '_tf32_' + name + '_vs_own_fp32_' + metric] = number
        rows.append(row)
    return rows


def triangle_records():
    return {s: read(ROOT / s / ('rounding_leak' if s == 'h100' else 'rounding') /
                    'G02/triangle_rounding.json')['records'] for s in ('ppu', 'h100')}


def query_drift_rows():
    data = {s: read(ROOT / s / 'key_positions.json') for s in ('ppu', 'h100')}
    source = hashlib.sha256((ROOT / 'check_triangle_key_positions.py').read_bytes()).hexdigest()
    assert all(v['source_sha256'] == source and v['status'] == 'EXECUTION_PASS' for v in data.values())
    result = []
    for p, h in zip(data['ppu']['records'], data['h100']['records']):
        assert p['key_position'] == h['key_position']
        row = {'key_position': p['key_position'], 'gold_score_and_natural_lse': 1.}
        for server, record in (('ppu', p), ('h100', h)):
            for mode, value in record['modes'].items():
                row[server + '_' + mode + '_natural_lse'] = value['lse_mean']
                row[server + '_' + mode + '_output_vs_gold_relative_l2'] = value['output_vs_gold']['relative_l2']
                assert value['output_vs_gold']['finite']
        expected = 1. + ((h['key_position'] // 16 + 1) // 2) / 1024
        assert h['modes']['tf32']['lse_mean'] == h['modes']['tf32']['maximum_mean'] == expected
        assert h['modes']['fp32']['lse_mean'] == 1.
        assert all(abs(p['modes'][mode]['lse_mean'] - 1.) < 1e-6 for mode in ('fp32', 'tf32'))
        row['native_repeated_rounding_score_prediction'] = expected
        row['scope'] = 'Identical exact scalar operands; one valid key. PPU base-2 cache converted using source-verified units.'
        result.append(row)
    assert len(result) == 16
    return result


def leaky_query_rows():
    result = []
    for record in triangle_records()['h100']:
        for name, value in record['leaky_query_decomposition'].items():
            row = {'family': record['family'], 'decomposition': name,
                'native_tf32_vs_own_ieee_relative_l2': record['replays']['tf32']['vs_unmodified_fp32']['relative_l2'],
                'reproduction_vs_native_tf32_relative_l2': record['replays'][name]['vs_unmodified_tf32']['relative_l2'],
                **{kind + '_relative_l2': value[kind]['relative_l2']
                   for kind in ('lse_vs_native_tf32', 'maximum_vs_native_tf32')}}
            assert row['maximum_vs_native_tf32_relative_l2'] < 2e-7
            assert row['lse_vs_native_tf32_relative_l2'] < 2e-7
            result.append(row)
    return result


def exact_input_rows():
    data = {s: read(ROOT / s / 'exact_inputs.json') for s in ('ppu', 'h100')}
    source = hashlib.sha256((ROOT / 'check_triangle_exact_inputs.py').read_bytes()).hexdigest()
    assert all(v['source_sha256'] == source and v['status'] == 'PASS' for v in data.values())
    assert len(data['ppu']['records']) == len(data['h100']['records']) == 4
    rows = []
    for p, h in zip(data['ppu']['records'], data['h100']['records']):
        for key in ('query_scalar', 'scale', 'scale_fp32', 'canonical_inputs', 'gold_qk_max_natural'):
            assert p[key] == h[key], key
        row = {key: h[key] for key in ('query_scalar', 'scale', 'scale_fp32', 'gold_qk_max_natural')}
        row['canonical_input_sha256'] = str([v['sha256'] for v in h['canonical_inputs']])
        row['scope'] = 'Matched constructed inputs; full CPU FP64 softmax gold; accuracy only, not model inputs or latency'
        for server, record in (('ppu', p), ('h100', h)):
            for mode, value in record['modes'].items():
                errors = value['vs_fp64_softmax_gold']
                assert errors['finite'] and all(math.isfinite(errors[k]) for k in ('relative_l2', 'rms', 'max_abs'))
                for key in ('relative_l2', 'rms', 'max_abs'):
                    row[server + '_' + mode + '_vs_fp64_' + key] = errors[key]
                for key in ('maximum_mean', 'lse_mean', 'kernel_trace'):
                    if key in value:
                        row[server + '_' + mode + '_' + key] = str(value[key]) if key == 'kernel_trace' else value[key]
        row['h100_tf32_vs_fp32_auxiliary_maximum_gain'] = h['tf32_vs_fp32_maximum_gain']
        rows.append(row)
    return rows


def main():
    screen = read(ROOT / 'ppu/screen_derived.json')
    p = read(ROOT / 'ppu/validation_derived.json')
    h = read(ROOT / 'h100/clean_native_derived.json')
    pp, hh, ss = [v['cases']['G02'] for v in (p, h, screen)]
    assert p['checkpoint']['sha256'] == h['checkpoint']['sha256'] == screen['checkpoint']['sha256']
    for other in (hh, ss):
        for key in ('features', 'atom_count', 'atom_identity_sha256', 'canonical_input_sha256', 'asset_hashes'):
            assert pp['input_audit'][key] == other['input_audit'][key], key
        for key in ('initialization_semantic_sha256', 'prediction_semantic_sha256'):
            assert pp['rng_audit'][key] == other['rng_audit'][key]
    assert pp['environment']['use_flag_gems'] and pp['environment']['flag_gems_exclude'] == []
    assert not hh['environment']['use_flag_gems']
    old_p = read(ROOT.parent / 'gemv_qk_20261002/ppu/column_model/G02/paired.json')['summary']
    old_h = read(ROOT.parent / 'gemv_qk_20261002/h100/model/G02/paired.json')['summary']
    for mode in ('fp32', 'tf32'):
        assert pp['summary'][mode]['prediction_digest'] == old_p['gemv_tf32' if mode == 'tf32' else mode]['prediction_digest']
        assert hh['summary'][mode]['prediction_digest'] == old_h[mode]['prediction_digest']
        assert ss['summary'][mode]['prediction_digest'] == pp['summary'][mode]['prediction_digest']
    anchors = read(ROOT.parent / 'rank_one_20261002/previous_matched_anchors.json')['structure']
    fp_anchor = next(x for x in anchors if x['input'] == 'G02' and x['mode'] == 'ppu_fp32')
    tf_anchor = next(x for x in anchors if x['input'] == 'G02' and x['mode'] == 'ppu_tf32')
    assert hh['drift_vs_frozen']['tf32']['rmsd_vs_frozen_same_precision_A'] < 1e-10
    baseline = {s + '_' + m: case['summary'][m] for s, case in (('h100', hh), ('ppu', pp))
                for m in ('fp32', 'tf32')}
    four_modes = {'model': 'protenix_base_default_v1.0.0', 'input': 'G02', 'N_token': 245,
                  'cycle': 1, 'step': 1, 'sample': 1, 'warmups_per_mode': 3, 'repeats_per_mode': 10}
    four_modes.update({m + '_seconds': v['median_seconds'] for m, v in baseline.items()})
    confidence = [{'metric': metric, **{m: v['five_confidence_metrics'][metric] for m, v in baseline.items()}}
                  for metric in METRICS]
    structure = []
    for mode, value in ss['summary'].items():
        reference = hh['summary']['fp32' if mode == 'fp32' else 'tf32']
        if mode == 'fp32':
            lo, hi = fp_anchor['rmsd_lower_bound_A'], fp_anchor['rmsd_upper_bound_A']
        else:
            drift = ss['drift_vs_frozen'][mode]['rmsd_vs_frozen_same_precision_A']
            c, t, e = distance(ss, mode, 'fp32'), distance(hh, 'fp32', 'tf32'), fp_anchor['rmsd_upper_bound_A']
            lo = max(0, tf_anchor['rmsd_exact_A'] - drift, abs(c - t) - e)
            hi = min(tf_anchor['rmsd_exact_A'] + drift, c + t + e)
        assert 0 <= lo <= hi
        errors = {metric: 100 * abs(value['five_confidence_metrics'][metric] - reference['five_confidence_metrics'][metric])
                  / abs(reference['five_confidence_metrics'][metric]) for metric in METRICS}
        row = {'mode': mode, 'reference': 'h100_' + ('fp32' if mode == 'fp32' else 'tf32'),
               'rmsd_lower_bound_A': lo, 'rmsd_upper_bound_A': hi,
               'rmsd_gate_0_1A': 'PASS' if hi <= .1 else 'FAIL' if lo > .1 else 'UNRESOLVED',
               'five_confidence_gate_5pct': all(v <= 5 for v in errors.values()),
               'same_ppu_rmsd_vs_fp32_A': distance(ss, mode, 'fp32') if mode != 'fp32' else 0,
               'screen_seconds_1warmup_3repeat': value['median_seconds'],
               **{m + '_seconds_3warmup_10repeat': v['median_seconds'] for m, v in baseline.items()},
               **{metric + '_relative_percent': v for metric, v in errors.items()},
               'method': 'Rigorous frozen TF32 and strict FP32 anchor triangle intersection; not direct cross-server RMSD'}
        structure.append(row)
    boundaries = boundary_rows('boundaries_compact.json')
    granular = boundary_rows('granular_compact.json') if all((ROOT / s / 'granular_compact.json').exists() for s in ('ppu', 'h100')) else []
    rounding = triangle_rows() if all((ROOT / s / 'rounding/G02/triangle_rounding.json').exists() for s in ('ppu', 'h100')) else []
    auxiliary = auxiliary_rows() if rounding else []
    exact_inputs = exact_input_rows()
    scalar_map = read(ROOT / 'h100/scalar_map.json')
    assert scalar_map['source_sha256'] == hashlib.sha256((ROOT / 'check_triangle_scalar_map.py').read_bytes()).hexdigest()
    key_positions = query_drift_rows()
    query_leak = leaky_query_rows()
    binary_audit = read(ROOT / 'h100/binary_audit/native_binary_audit.json')
    assert binary_audit['source_sha256'] == hashlib.sha256((ROOT / 'inspect_native_triangle_binary.py').read_bytes()).hexdigest()
    diagnostic_audits = {}
    for server, baseline_case in (('ppu', pp), ('h100', hh)):
        for folder in ('derived', 'granular_derived', 'rounding_derived', 'rounding_leak_derived'):
            path = ROOT / server / (folder + '.json')
            if not path.exists():
                continue
            evidence = read(path)['cases']['G02']
            for mode in ('fp32', 'tf32'):
                assert evidence['summary'][mode]['prediction_digest'] == baseline_case['summary'][mode]['prediction_digest']
                assert evidence['summary'][mode]['five_confidence_metrics'] == baseline_case['summary'][mode]['five_confidence_metrics']
            for key in ('features', 'atom_identity_sha256', 'canonical_input_sha256', 'asset_hashes'):
                assert evidence['input_audit'][key] == baseline_case['input_audit'][key]
            for key in ('initialization_semantic_sha256', 'prediction_semantic_sha256'):
                assert evidence['rng_audit'][key] == baseline_case['rng_audit'][key]
            diagnostic_audits[server + '_' + folder] = {'status': 'PASS', 'rng': evidence['rng_audit']}
    first = next(row for row in boundaries if row['kind'] == 'early_linear' and row['side'] == 'outputs'
                 and (row['tf32_cross_device_relative_l2_estimate'] or 0) >= 1e-4
                 and (row['fp32_cross_device_relative_l2_estimate'] or 0) <= 1e-5)
    protected = ROOT.parent.parent / 'ppu_mean_fix_20260920/benchmark_e2e.py'
    assert hashlib.sha256(protected.read_bytes()).hexdigest() == '6ede90f98d9435b8509fb37b086c6a490296b49a99c6fbbd4094eda2f5506d42'
    report = {'status': 'NATIVE_TRIANGLE_QUERY_ROUNDING_DRIFT_REPRODUCED', 'inference': four_modes,
              'confidence': confidence, 'structure': structure, 'first_sampled_linear_divergence': first,
              'boundaries': boundaries, 'granular_boundaries': granular,
              'triangle_replay': rounding, 'diagnostic_audits': diagnostic_audits,
              'triangle_auxiliary': auxiliary,
              'triangle_exact_inputs': exact_inputs, 'native_triangle_scalar_map': scalar_map,
              'triangle_key_position_controls': key_positions, 'native_query_leak_decomposition': query_leak,
              'native_binary_audit': binary_audit, 'native_counterfactual_low_bits': read(ROOT / 'h100/low_bits.json'),
              'findings': [
                  'Installed native CUE 0.8.0 TF32 query score depends on key position even for exact TF32 inputs. FP32 and public PPU FlagGems controls have no measured drift.',
                  'For q=k=1 with one valid key, native score and LSE match 1 + floor((floor(key_index/16)+1)/2)/1024 exactly for all sixteen measured positions.',
                  'Repeated unmasked integer addition of 0x1000 to query bit state per 16 keys reproduces native real-activation maximum/LSE with relative L2 below 2e-7 in three nonzero module families.',
                  'Matching the observed native query drift reduces full attention output mismatch substantially; remaining probability/V accumulation effects are not resolved.',
                  'No accepted PPU policy or end-to-end acceleration resulted from this diagnostic round. The frozen native TF32 baseline has not been replaced.',
                  'This operator diagnosis does not yet quantify the full end-to-end RMSD contribution of the native query drift.'],
              'next_actions': [
                  'Validate a one-time TF32 query conversion that preserves immutable Q across KV tiles in an isolated implementation; verify score permutation behavior, outputs and backward.',
                  'Check an isolated newer native CUE build with the same reproducer; keep the official frozen four-mode baseline separately identifiable.',
                  'Resolve native-reference correctness before selecting high-precision restores from closeness to that reference; retain same-precision and strict-FP32 comparisons.',
                  'Resume general PPU kernel scheduling/PV optimization under a verified precision contract, then nine-model multi-input and training acceptance.'],
              'roles': read(ROOT / 'ppu/screen/G02/paired.json')['roles'],
              'restored_calls': read(ROOT / 'ppu/screen/G02/paired.json')['restored_calls_first_warmup'],
              'checkpoint': p['checkpoint'], 'input_audit': pp['input_audit'],
              'rng_audits': {'ppu_validation': pp['rng_audit'], 'ppu_screen': ss['rng_audit'], 'h100': hh['rng_audit']},
              'environment': {'ppu': pp['environment'], 'h100': hh['environment']},
              'source_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.iterdir() if f.suffix in ('.py', '.sh')},
              'limits': ['Single base model, G02, one cycle/step/sample; nine-model, multi-input and full training not rerun.',
                         'All seven changed TF32 policies fail the 0.1 A structure gate; no policy promoted.',
                         'Boundary cross-device L2 uses aggregate estimates, not exact tensor distance or acceptance.',
                         'Full same-device TF32/FP32 activation error is exact; event order is pre-hook order, not output completion order.',
                         'Original boundary-run timings excluded: inactive hooks remained registered; clean baseline rerun used.',
                         'Candidate screen has one warmup/three repeats; compare timing cautiously, no speedup claim.',
                         'Later rounding diagnostics found H100 GPU7 shared with another workload; those timings are excluded. Precision hashes remained unchanged.',
                         'First sampled linear is not a claim about first global ATen divergence.',
                         'Dense references use full operands and strict FP32 arithmetic; they are diagnostic decompositions, not FP64 gold or latency comparisons.',
                         'Single-key score and LSE gold controls establish native position-dependent drift; auxiliary maximum alone was insufficient to localize it.',
                         'Synthetic controls use CPU FP64 gold and matching canonical Q/K/V hashes, separately from the actual model activation replays.',
                         'Submodule relative errors have different output norms; absolute RMS estimates are included and no direct error amplification factor is inferred.',
                         'Only aggregate scalar/hash artifacts transferred; activation/weight/coordinate tensors stay on original servers.',
                         'No formal MFU or normalized 80% efficiency claim; default server packages unchanged.']}
    (ROOT / 'complete_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    write_csv('four_mode_inference.csv', [four_modes])
    write_csv('four_mode_confidence.csv', confidence)
    write_csv('precision_factorial.csv', structure)
    write_csv('four_mode_boundaries.csv', boundaries)
    if granular:
        write_csv('four_mode_granular_boundaries.csv', granular)
    if rounding:
        write_csv('four_mode_triangle_replay.csv', rounding)
        write_csv('four_mode_triangle_auxiliary.csv', auxiliary)
    write_csv('four_mode_triangle_exact_inputs.csv', exact_inputs)
    write_csv('four_mode_key_position_controls.csv', key_positions)
    write_csv('native_query_drift_reproduction.csv', query_leak)
    print('FOUR_MODE_INPUT_RNG_CHECKPOINT_HASH_AUDIT_PASS', four_modes)
    print('PRECISION_GATES', [(r['mode'], r['rmsd_gate_0_1A'], r['rmsd_lower_bound_A'], r['rmsd_upper_bound_A']) for r in structure])


if __name__ == '__main__':
    main()
