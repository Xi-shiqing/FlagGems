"""Join scalar evidence with the original four modes and explicit candidates."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIDES = ('h100', 'ppu')
PAIRS = (('h100', 'fp32'), ('h100', 'tf32'), ('ppu', 'fp32'), ('ppu', 'tf32'))
COLUMNS = ('h100_native_fp32', 'h100_native_tf32', 'ppu_flaggems_fp32', 'ppu_flaggems_tf32')
NA = 'UNSUPPORTED_NATIVE_FP32_BACKWARD'


def read(path):
    return json.loads(path.read_text())


def load(name):
    return {side: read(ROOT / side / name) for side in SIDES}


def save_csv(name, rows):
    if rows:
        with (ROOT / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def timing(mode, phase, clock):
    value = mode['timing_median'][phase]
    return value[clock] if value else NA


def error(mode, kind):
    if kind == 'output':
        return mode['output']['relative_l2'] * 100
    if not mode['parameters']:
        return NA
    return max(value[kind]['relative_l2'] for value in mode['parameters'].values()) * 100


def index_cases(data):
    return {side: {tuple(case['shape']): case for case in value['cases']} for side, value in data.items()}


def main():
    training, forward = load('triangle_training_final.json'), load('forward_trial.json')
    live, forward_live = load('rounding_clean/G02/triangle_rounding.json'), load('forward_model/G02/triangle_rounding.json')
    audits, model_audits = load('derived_live_clean.json'), load('derived_forward_model.json')
    general, general_forward, largest = load('general_backward.json'), load('general_forward.json'), load('largest_update.json')
    pairs = load('forward_model/G02/paired.json')
    model_bench = load('forward_model/G02/benchmark.json')
    retained_audits = read(ROOT / 'retained_previous_audits.json')
    prior_audits = retained_audits['audits']
    for data in (training, forward):
        assert all(x['status'] == 'EXECUTION_PASS' for x in data.values())
    for data in (general, general_forward, largest, live, forward_live, pairs):
        assert all(x['status'] == 'PASS' for x in data.values())
    audit_checks = {}
    for name, data in (('backward_live', audits), ('forward_model', model_audits)):
        assert data['h100']['checkpoint']['sha256'] == data['ppu']['checkpoint']['sha256']
        assert all(x['checkpoint']['current_hash_rechecked'] for x in data.values())
        for key in ('initialization_semantic_sha256', 'prediction_semantic_sha256'):
            assert data['h100']['cases']['G02']['rng_audit'][key] == data['ppu']['cases']['G02']['rng_audit'][key]
        for key in ('canonical_input_sha256', 'atom_identity_sha256', 'asset_hashes', 'features'):
            assert data['h100']['cases']['G02']['input_audit'][key] == data['ppu']['cases']['G02']['input_audit'][key], key
        for side in SIDES:
            for mode, current in data[side]['cases']['G02']['summary'].items():
                previous = prior_audits[side]['cases']['G02']['summary']['fp32' if mode == 'fp32' else 'tf32']
                assert current['prediction_digest'] == previous['prediction_digest']
                assert current['five_confidence_metrics'] == previous['five_confidence_metrics']
        audit_checks[name] = {'checkpoint_input_assets_atoms_features_rng': 'PASS', 'predictions_and_confidence_equal_previous_round': True,
                             'legacy_vendor_anchor_scope': 'Stored drift_vs_frozen refers to older vendor anchors, not the current matched baseline; those existing nonzero drifts are retained explicitly.'}
    for data in (general, general_forward):
        assert len(data['h100']['cases']) == len(data['ppu']['cases'])
        for a, b in zip(data['h100']['cases'], data['ppu']['cases']):
            assert a['shape'] == b['shape'] and a['input_sha256'] == b['input_sha256']
    assert largest['h100']['input_sha256'] == largest['ppu']['input_sha256']
    cases, fw_cases = index_cases(training), index_cases(forward)
    assert cases['h100'].keys() == cases['ppu'].keys() == fw_cases['h100'].keys() == fw_cases['ppu'].keys()
    timing_rows, accuracy_rows, memory_rows, profiles = [], [], [], []
    for shape in cases['ppu']:
        for data in (cases, fw_cases):
            assert data['h100'][shape]['full_input_sha256'] == data['ppu'][shape]['full_input_sha256']
        assert cases['ppu'][shape]['full_input_sha256'] == fw_cases['ppu'][shape]['full_input_sha256']
        for phase, data, candidate in (('training', cases, 'bounded_prequant'), ('inference', fw_cases, 'forward_kv_prequant')):
            modes = {side: data[side][shape]['modes'] for side in SIDES}
            for side in SIDES:
                assert modes[side][candidate]['output_sha256'] == modes[side]['fixed_rne_once']['output_sha256']
            assert modes['ppu'][candidate]['output_sha256'] == modes['ppu']['tf32']['output_sha256']
            for clock in ('event_ms', 'wall_ms'):
                row = {'B': shape[0], 'N': shape[1], 'S': shape[2], 'H': shape[3], 'D': shape[4],
                       'phase': phase, 'clock': clock, 'warmups': 3, 'repeats': 10, 'candidate': candidate}
                row.update({column: timing(modes[side][mode], phase, clock) for column, (side, mode) in zip(COLUMNS, PAIRS)})
                row['ppu_flaggems_candidate_tf32'] = timing(modes['ppu'][candidate], phase, clock)
                row['h100_custom_parent_rne_tf32'] = timing(modes['h100']['fixed_rne_once'], phase, clock)
                row['h100_custom_candidate_tf32'] = timing(modes['h100'][candidate], phase, clock)
                row['ppu_candidate_speedup_vs_ppu_tf32'] = row['ppu_flaggems_tf32'] / row['ppu_flaggems_candidate_tf32']
                row['ppu_candidate_time_saving_percent'] = 100 * (1 - row['ppu_flaggems_candidate_tf32'] / row['ppu_flaggems_tf32'])
                row['ppu_old_time_over_h100_native_same_precision'] = row['ppu_flaggems_tf32'] / row['h100_native_tf32']
                row['ppu_candidate_time_over_h100_native_same_precision'] = row['ppu_flaggems_candidate_tf32'] / row['h100_native_tf32']
                timing_rows.append(row)
            for metric in ('output', 'gradient', 'updated_parameter', 'realized_fp32_update'):
                row = {'S': shape[2], 'candidate': candidate, 'metric': metric, 'unit': 'relative_L2_percent',
                       'gold': 'CPU FP64; maximum across full Q/K/V/bias2, except output; updated parameter rounded once to FP32'}
                row.update({column: error(modes[side][mode], metric) for column, (side, mode) in zip(COLUMNS, PAIRS)})
                row.update(ppu_flaggems_candidate_tf32=error(modes['ppu'][candidate], metric),
                           h100_custom_parent_rne_tf32=error(modes['h100']['fixed_rne_once'], metric),
                           h100_custom_candidate_tf32=error(modes['h100'][candidate], metric))
                accuracy_rows.append(row)
            row = {'S': shape[2], 'candidate': candidate, 'unit': 'bytes',
                   'scope': 'peak extra allocated GPU memory for one forward/backward accuracy call, not inference-only or whole-model peak'}
            row.update({column: modes[side][mode]['accuracy_call_peak_extra_bytes'] for column, (side, mode) in zip(COLUMNS, PAIRS)})
            row['h100_native_fp32'] = NA  # No supported full backward call to compare.
            row.update(ppu_flaggems_candidate_tf32=modes['ppu'][candidate]['accuracy_call_peak_extra_bytes'],
                       h100_custom_parent_rne_tf32=modes['h100']['fixed_rne_once']['accuracy_call_peak_extra_bytes'],
                       h100_custom_candidate_tf32=modes['h100'][candidate]['accuracy_call_peak_extra_bytes'])
            memory_rows.append(row)
        for side in SIDES:
            for name, mode in cases[side][shape]['modes'].items():
                for kernel in mode.get('training_gpu_profile', []):
                    profiles.append({'server': side, 'S': shape[2], 'mode': name, **kernel,
                                     'scope': 'one profiled call after clean timing; kernel totals are diagnostic'})
    live_rows, live_accuracy, live_hashes = [], [], []
    for trial, data, candidate in (('backward', live, 'bounded_prequant'), ('forward', forward_live, 'forward_kv_prequant')):
        records = {side: {r['family']: r for r in value['records']} for side, value in data.items()}
        assert records['ppu'].keys() == records['h100'].keys()
        for family in records['ppu']:
            a, b = records['h100'][family]['live_backward'], records['ppu'][family]['live_backward']
            assert a['seed'] == b['seed'] and a['upstream_sha256'] == b['upstream_sha256']
            assert 'suspended' in a['timing_instrumentation'] and 'suspended' in b['timing_instrumentation']
            live_hashes.append({'trial': trial, 'family': family, 'matched_cpu_upstream': True,
                               'activation_hashes': {side: records[side][family]['live_backward']['raw_parameter_inputs'] for side in SIDES},
                               'scope': 'each device uses its own native TF32 teacher activations; activations intentionally differ across devices'})
            modes = {side: records[side][family]['live_backward']['modes'] for side in SIDES}
            for phase in ('inference', 'training'):
                for clock in ('event_ms', 'wall_ms'):
                    row = {'trial': trial, 'family': family, 'phase': phase, 'clock': clock, 'warmups': 3, 'repeats': 10,
                           'scope': 'whole isolated operator on own teacher activations; confidence_head is a zero-Q/K control'}
                    row.update({column: timing(modes[side][mode], phase, clock) for column, (side, mode) in zip(COLUMNS, PAIRS)})
                    row.update(ppu_flaggems_candidate_tf32=timing(modes['ppu'][candidate], phase, clock),
                               h100_custom_parent_rne_tf32=timing(modes['h100']['custom_parent_rne'], phase, clock),
                               h100_custom_candidate_tf32=timing(modes['h100'][candidate], phase, clock))
                    row['ppu_candidate_time_saving_percent'] = 100 * (1 - row['ppu_flaggems_candidate_tf32'] / row['ppu_flaggems_tf32'])
                    live_rows.append(row)
            for metric in ('output_sample_vs_fp64', 'gradient_sample_vs_fp64', 'gradient_vs_parent_rne', 'realized_update_vs_parent_rne'):
                def live_error(mode):
                    if metric == 'output_sample_vs_fp64':
                        return mode[metric]['relative_l2'] * 100
                    values = [v[metric]['relative_l2'] for v in mode['parameters'].values() if metric in v]
                    return max(values) * 100 if values else NA
                row = {'trial': trial, 'family': family, 'metric': metric, 'unit': 'relative_L2_percent',
                       'scope': 'FP64 Q/K/V gold: first/last N, all S/head/D; bias compared fully against parent; own teacher activations'}
                row.update({column: live_error(modes[side][mode]) for column, (side, mode) in zip(COLUMNS, PAIRS)})
                row.update(ppu_flaggems_candidate_tf32=live_error(modes['ppu'][candidate]),
                           h100_custom_parent_rne_tf32=live_error(modes['h100']['custom_parent_rne']),
                           h100_custom_candidate_tf32=live_error(modes['h100'][candidate]))
                live_accuracy.append(row)
            for side in SIDES:
                row = modes[side][candidate]
                assert row['output_vs_parent_rne']['max_abs'] == 0
                assert all(v['gradient_vs_parent_rne']['max_abs'] == 0 and v['realized_update_vs_parent_rne']['max_abs'] == 0 for v in row['parameters'].values())
    model_rows, confidence_rows = [], []
    summaries = {side: pairs[side]['summary'] for side in SIDES}
    model_row = {'model': 'protenix_base_default_v1.0.0', 'input': 'G02', 'tokens': 245, 'atoms': 2529,
                 'cycle': 1, 'step': 1, 'sample': 1, 'seed': 101, 'warmups_per_mode': 3, 'repeats_per_mode': 10, 'unit': 'seconds'}
    model_row.update({column: summaries[side][mode]['median_seconds'] for column, (side, mode) in zip(COLUMNS, PAIRS)})
    model_row['ppu_flaggems_forward_kv_tf32'] = summaries['ppu']['forward_kv_tf32']['median_seconds']
    model_row['ppu_candidate_time_saving_percent'] = 100 * (1 - model_row['ppu_flaggems_forward_kv_tf32'] / model_row['ppu_flaggems_tf32'])
    model_row['ppu_candidate_time_over_h100_native_tf32'] = model_row['ppu_flaggems_forward_kv_tf32'] / model_row['h100_native_tf32']
    old_samples, new_samples = [summaries['ppu'][mode]['samples_seconds'] for mode in ('tf32', 'forward_kv_tf32')]
    model_row['ppu_tf32_mean_seconds'] = statistics.mean(old_samples)
    model_row['ppu_candidate_mean_seconds'] = statistics.mean(new_samples)
    model_row['ppu_candidate_faster_paired_repeats'] = sum(b < a for a, b in zip(old_samples, new_samples))
    model_row['ppu_paired_median_time_saving_ms'] = statistics.median(1000 * (a - b) for a, b in zip(old_samples, new_samples))
    model_row['speed_gain_status'] = 'NO_RELIABLE_END_TO_END_GAIN: median difference smaller than run variation; mean and paired median do not improve'
    model_rows.append(model_row)
    for metric in ('plddt', 'gpde', 'ptm', 'iptm', 'ranking_score'):
        row = {'model': model_row['model'], 'input': 'G02', 'metric': metric}
        row.update({column: summaries[side][mode]['five_confidence_metrics'][metric] for column, (side, mode) in zip(COLUMNS, PAIRS)})
        row['ppu_flaggems_forward_kv_tf32'] = summaries['ppu']['forward_kv_tf32']['five_confidence_metrics'][metric]
        confidence_rows.append(row)
    assert pairs['ppu']['candidate_calls'] and all(x['precision'] == 'tf32' for x in pairs['ppu']['candidate_calls'])
    assert summaries['ppu']['forward_kv_tf32']['prediction_digest'] == summaries['ppu']['tf32']['prediction_digest']
    model_memory, structure_rows = [], []
    for metric in ('peak_memory_allocated_bytes', 'peak_memory_reserved_bytes'):
        def peak(side, mode):
            values = [r[metric] for r, m in zip(model_bench[side]['records'], pairs[side]['order']) if m == mode and r['phase'] == 'repeat']
            assert len(values) == 10
            return max(values)
        row = {'input': 'G02', 'metric': metric, 'aggregation': 'max over 10 model repeats', 'unit': 'bytes'}
        row.update({column: peak(side, mode) for column, (side, mode) in zip(COLUMNS, PAIRS)})
        row['ppu_flaggems_forward_kv_tf32'] = peak('ppu', 'forward_kv_tf32')
        model_memory.append(row)
    for reference in ('fp32', 'tf32'):
        def distance(side, mode):
            if summaries[side][mode]['prediction_digest'] == summaries[side][reference]['prediction_digest']:
                return 0.0
            values = model_audits[side]['cases']['G02']['same_device_rmsd_A']
            key = mode + '_vs_' + reference
            return values[key] if key in values else values[reference + '_vs_' + mode]
        row = {'input': 'G02', 'metric': 'RMSD_vs_own_' + reference, 'unit': 'Angstrom',
               'scope': 'own-device reference; identical coordinate hashes imply exact zero, raw Kabsch numerical residue retained in derived audit'}
        row.update({column: distance(side, mode) for column, (side, mode) in zip(COLUMNS, PAIRS)})
        row['ppu_flaggems_forward_kv_tf32'] = distance('ppu', 'forward_kv_tf32')
        structure_rows.append(row)
    ablations = []
    for source in ('triangle_training.json', 'triangle_training_final.json'):
        for side, data in load(source).items():
            for case in data['cases']:
                for mode, value in case['modes'].items():
                    ablations.append({'source': source, 'server': side, 'S': case['shape'][2], 'mode': mode,
                        'training_event_ms': timing(value, 'training', 'event_ms'), 'inference_event_ms': timing(value, 'inference', 'event_ms'),
                        'max_gradient_relative_L2_percent': error(value, 'gradient'),
                        'decision': 'rejected: changes precision' if mode == 'round_off_diagnostic' else
                                    ('not general default: slower at smaller sizes' if mode == 'prequant_cached_ds' else 'see candidate decision')})
    tables = {'four_mode_operator_timing.csv': timing_rows, 'four_mode_operator_accuracy.csv': accuracy_rows,
              'four_mode_live_activation_timing.csv': live_rows, 'four_mode_live_activation_accuracy.csv': live_accuracy,
              'four_mode_model_inference.csv': model_rows, 'four_mode_confidence.csv': confidence_rows,
              'four_mode_model_memory.csv': model_memory, 'four_mode_model_structure.csv': structure_rows,
              'four_mode_operator_memory.csv': memory_rows, 'backward_kernel_profile.csv': profiles, 'diagnostic_ablations.csv': ablations}
    for name, rows in tables.items():
        save_csv(name, rows)
    report = {'status': 'GENERIC_TF32_BACKWARD_VALIDATED_FORWARD_MODEL_VALIDATED_G02',
        'production_change': False, 'default_native_package_change': False, 'commit_or_push': False,
        'four_mode_columns': COLUMNS, 'candidate_decision': {
            'bounded_prequant': 'Preferred backward candidate: one global RNE of Q/K/V/dO per N chunk; raw dO retained for delta; P/dS rounding preserved. No new exact-shape whitelist.',
            'forward_kv_prequant': 'Forward trial: one global RNE of K/V; Q/P rounding preserved. Generic 256 MiB converted-input eligibility, otherwise original forward. Operator improves, but no reliable G02 end-to-end speed gain: retain experiment, no default promotion.',
            'round_off_diagnostic': 'Reject: faster but raises gradient error; precision-changing lower-bound ablation only.',
            'prequant_cached_ds': 'Do not promote as universal default: faster at largest synthetic size, slower than preferred at smaller sizes and on live activations.',
            'h100_custom': 'Diagnostic custom kernels only, never replace native FP32/TF32 acceptance references.'},
        'synthetic_operand_scope': 'CPU PCG64 deterministic integers; three sizes repeat one seeded N group, both full and group hashes matched across devices. Separate correctness suite uses independent N groups, masks, batches, H2/4/8, D16/32/64, strided BNHSD, forced backward budget/chunk/fallback tests.',
        'timing_contract': '3 warmups/10 alternating repeats per mode; GPU event and synchronized wall separate. Full public operator entry, including native CUE fused return_aux and internal packing; operator training includes forward/backward/SGD on leaf Q/K/V/bias, not whole-model training. Model timings are separate 3/10 synchronized repeats.',
        'native_fp32_backward': 'CUE 0.8.0 explicitly unsupported: N/A; no dense substitute.',
        'memory_contract': 'Backward 256 MiB bounds converted-input cache only, not total peak or chunk gradients. Forward 256 MiB eligibility bounds converted K/V only. No total peak/OOM guarantee.',
        'tables': tables, 'general_backward_status': {side: data['status'] for side, data in general.items()},
        'general_forward_status': {side: data['status'] for side, data in general_forward.items()},
        'largest_chunk_comparison': {side: data['modes']['bounded_prequant'] for side, data in largest.items()},
        'live_input_hash_audits': live_hashes, 'audit_checks': audit_checks,
        'audits': {name: {side: {'checkpoint': data['checkpoint'], **{key: data['cases']['G02'][key] for key in ('rng_audit', 'input_audit', 'drift_vs_frozen')}} for side, data in source.items()}
                   for name, source in (('backward_live', audits), ('forward_model', model_audits))},
        'candidate_real_model_call_count': len(pairs['ppu']['candidate_calls']),
        'retained_baseline_source_sha256': retained_audits['source_sha256'],
        'end_to_end_speed_decision': model_row['speed_gain_status'],
        'confidence_scope': 'Actual candidate predictions exactly equal local TF32 baseline in all warmup/repeat digests and all five confidence metrics; coordinates stay on original server. Same-device candidate RMSD is 0.',
        'cross_device_structure_scope': 'Original PPU TF32 vs native H100 TF32 interval remains [0.086655040494,0.218147146445] Angstrom, unresolved at 0.1 Angstrom threshold. No direct cross-device coordinates exported.',
        'remaining': ['Combine forward/backward candidates in one operator and validate gradients at real-model training entry.',
                      'Nine-model multi-input candidate acceptance remains outstanding.',
                      'Generic cache budgeting and total memory peak need further work before default promotion.',
                      'No absolute MFU reported without verified current PPU peak throughput.'],
        'source_evidence_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for side in SIDES for path in sorted((ROOT / side).rglob('*.json'))}}
    (ROOT / 'complete_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print('BACKWARD_FORWARD_COMPARISON_COMPLETE', {name: len(rows) for name, rows in tables.items()})


if __name__ == '__main__':
    main()
