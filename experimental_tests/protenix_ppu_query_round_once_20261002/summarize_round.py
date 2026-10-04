"""Join scalar-only evidence, retaining original native baselines explicitly."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLUMNS = ('h100_native_fp32', 'h100_native_tf32', 'ppu_flaggems_fp32', 'ppu_flaggems_tf32')
PAIRS = (('h100', 'fp32'), ('h100', 'tf32'), ('ppu', 'fp32'), ('ppu', 'tf32'))
CANDIDATES = ('fixed_rne_once', 'fixed_rne_bits_once', 'fixed_rna_bits_once')


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    with (ROOT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def error(row, kind):
    if kind == 'output':
        return row['output']['relative_l2'] * 100
    if not row['parameters']:
        return 'UNSUPPORTED_NATIVE_FP32_BACKWARD'
    return max(value[kind]['relative_l2'] for value in row['parameters'].values()) * 100


def main():
    training = {side: read(ROOT / side / 'triangle_training.json') for side in ('h100', 'ppu')}
    audits = {side: read(ROOT / side / 'derived_live.json') for side in ('h100', 'ppu')}
    live = {side: read(ROOT / side / 'rounding/G02/triangle_rounding.json') for side in ('h100', 'ppu')}
    assert all(value['status'] == 'EXECUTION_PASS' for value in training.values())
    assert audits['h100']['checkpoint']['sha256'] == audits['ppu']['checkpoint']['sha256']
    for key in ('initialization_semantic_sha256', 'prediction_semantic_sha256'):
        assert audits['h100']['cases']['G02']['rng_audit'][key] == audits['ppu']['cases']['G02']['rng_audit'][key]
    for key in ('canonical_input_sha256', 'atom_identity_sha256', 'asset_hashes', 'features'):
        assert audits['h100']['cases']['G02']['input_audit'][key] == audits['ppu']['cases']['G02']['input_audit'][key], key
    cases = {side: {case['shape'][1]: case for case in value['cases']} for side, value in training.items()}
    assert cases['h100'].keys() == cases['ppu'].keys()
    timing_rows, accuracy_rows, candidate_rows, profiles = [], [], [], []
    for size in cases['h100']:
        assert cases['h100'][size]['group_input_sha256'] == cases['ppu'][size]['group_input_sha256']
        for phase in ('inference', 'training'):
            for clock in ('event_ms', 'wall_ms'):
                row = {'B': 1, 'N': size, 'S': size, 'H': 4, 'D': 32, 'phase': phase, 'clock': clock, 'warmups': 3, 'repeats': 10}
                for column, (side, mode) in zip(COLUMNS, PAIRS):
                    timing = cases[side][size]['modes'][mode]['timing_median'][phase]
                    row[column] = timing[clock] if timing else 'UNSUPPORTED_NATIVE_FP32_BACKWARD'
                timing_rows.append(row)
        for metric in ('output', 'gradient', 'updated_parameter', 'realized_fp32_update'):
            row = {'S': size, 'metric': metric, 'unit': 'relative_L2_percent', 'gold': 'CPU FP64; updated_parameter rounded once to FP32', 'aggregation': 'full output or maximum across Q/K/V/bias2'}
            row.update({column: error(cases[side][size]['modes'][mode], metric) for column, (side, mode) in zip(COLUMNS, PAIRS)})
            accuracy_rows.append(row)
        for side in ('h100', 'ppu'):
            original = cases[side][size]['modes']['tf32']
            for name in CANDIDATES:
                value = cases[side][size]['modes'][name]
                candidate_rows.append({'server': side, 'S': size, 'candidate': name, 'classification': 'custom kernel, not native',
                    'forward_relative_L2_percent': error(value, 'output'), 'max_gradient_relative_L2_percent': error(value, 'gradient'),
                    'max_realized_update_relative_L2_percent': error(value, 'realized_fp32_update'),
                    'output_hash_equal_original_tf32': value['output_sha256'] == original['output_sha256'],
                    'all_gradient_hashes_equal_original_tf32': all(value['parameters'][key]['gradient_sha256'] == original['parameters'][key]['gradient_sha256'] for key in value['parameters']),
                    'inference_event_ms': value['timing_median']['inference']['event_ms'],
                    'training_event_ms': value['timing_median']['training']['event_ms'],
                    'inference_time_vs_original_tf32': value['timing_median']['inference']['event_ms'] / original['timing_median']['inference']['event_ms'],
                    'training_time_vs_original_tf32': value['timing_median']['training']['event_ms'] / original['timing_median']['training']['event_ms']})
            for mode in ('fp32', 'tf32'):
                value = cases[side][size]['modes'][mode]
                kernels = value.get('training_gpu_profile', [])
                total = sum(kernel['total_us'] for kernel in kernels)
                for kernel in kernels:
                    profiles.append({'server': side, 'precision': mode, 'S': size, **kernel, 'share_percent_of_profiled_gpu_kernels': 100 * kernel['total_us'] / total if total else 0,
                                     'scope': 'One diagnostic training call after clean timing; not an end-to-end module share'})
    live_rows = []
    records = {side: {row['family']: row for row in value['records']} for side, value in live.items()}
    assert records['h100'].keys() == records['ppu'].keys()
    for family in records['h100']:
        row = {'family': family, 'metric': 'sample_relative_L2_percent_vs_CPU_FP64',
               'scope': 'Own TF32 teacher activations, all keys/full D for 2 groups and 3 queries; activations differ across devices',
               'zero_operand_control': family == 'confidence_head'}
        for column, (side, mode) in zip(COLUMNS, PAIRS):
            row[column] = records[side][family]['replays'][mode]['sample_vs_fp64']['relative_l2'] * 100
        for side in ('h100', 'ppu'):
            for name in CANDIDATES:
                row[side + '_custom_' + name] = records[side][family]['replays'][name]['sample_vs_fp64']['relative_l2'] * 100
        live_rows.append(row)
    write_csv('four_mode_operator_timing.csv', timing_rows)
    write_csv('four_mode_operator_accuracy.csv', accuracy_rows)
    write_csv('isolated_candidate_comparison.csv', candidate_rows)
    write_csv('four_mode_live_activation_accuracy.csv', live_rows)
    write_csv('training_kernel_profile.csv', profiles)
    prior = ROOT.parent / 'module_boundary_20261002'
    for name in ('four_mode_inference.csv', 'four_mode_confidence.csv'):
        (ROOT / ('retained_' + name)).write_bytes((prior / name).read_bytes())
    bits = {side: read(ROOT / side / 'rounding_bits.json') for side in ('h100', 'ppu')}
    assert bits['h100']['input_sha256'] == bits['ppu']['input_sha256']
    assert all(value['status'] == 'PASS' for value in bits.values())
    controls = {side: {name: read(ROOT / side / ('key_positions_' + name + '.json')) for name in CANDIDATES}
                for side in ('h100', 'ppu')}
    assert all(value['status'] == 'EXECUTION_PASS' for modes in controls.values() for value in modes.values())
    old_positions = read(prior / 'h100/key_positions.json')
    latest = read(ROOT / 'h100/key_positions_vendor0120.json')
    assert [row['modes']['tf32']['lse_mean'] for row in old_positions['records']] == [row['modes']['tf32']['lse_mean'] for row in latest['records']]
    report = {'status': 'OPERATOR_VALIDATION_COMPLETE_NO_PPU_SPEED_PROMOTION',
        'production_change': False, 'default_native_package_change': False, 'commit_or_push': False,
        'four_mode_columns': list(COLUMNS), 'operator_shapes': [[1, size, size, 4, 32] for size in cases['h100']],
        'operator_operand_scope': 'CPU NumPy PCG64 deterministic integers; N repeats the same seeded group. Matched group hashes and construction on both devices. Synthetic operands, not captured model activations.',
        'operator_timing_contract': 'CUDA-event and synchronized-wall timings are separate. Native CUE uses return_aux=True to force its fused forward entry and includes its auxiliary computation/internal packing; PPU uses public BNSHD entry. Native layout views are prepared before timing, and internal native packing is included. This is a public-operator comparison, not isolated MMA throughput or model speedup.',
        'bit_pattern_validation': bits,
        'native_query_drift': {'versions_black_box_tested': ['0.8.0', '0.12.0'], 'latest_installed_in_isolated_target_only': True,
            'version_import_audit': read(ROOT / 'h100/native_versions.json'),
            'identical_16_position_drift': True, 'last_key_score': latest['records'][-1]['modes']['tf32']['lse_mean'],
            'gold_score': 1., 'custom_rne_and_rna_once_controls': 'All 16 key positions pass output and natural-LSE gold on both devices'},
        'native_fp32_backward': 'CUE 0.8.0 explicitly unsupported on all three shapes; N/A, no fallback substituted',
        'candidate_decision': {'q_rne_once': 'PPU outputs and gradients unchanged; clean timing gives no material speed gain. Retain isolated candidate only.',
            'bit_rne_once': 'Exact rounding checks pass, but slower on PPU; reject promotion.',
            'rna_once': 'No consistent PPU forward gain and not RNE-equivalent; reject promotion.',
            'custom_h100': 'Accuracy improves, but these kernels are diagnostic references. Original native FP32/TF32 baselines remain unchanged.'},
        'four_mode_timing': timing_rows, 'four_mode_accuracy': accuracy_rows, 'candidate_comparisons': candidate_rows,
        'live_activation_accuracy': live_rows,
        'model_input': {'model': 'protenix_base_default_v1.0.0', 'case': 'G02', 'cycle': 1, 'step': 1, 'sample': 1,
                        'tokens': 245, 'atoms': 2529, 'seed': 101, 'candidate_applied_to_predictions': False},
        'audits': {side: {'checkpoint': data['checkpoint'], **{key: data['cases']['G02'][key] for key in ('rng_audit', 'input_audit', 'drift_vs_frozen')}} for side, data in audits.items()},
        'retained_end_to_end_scope': 'Previous 3-warmup/10-repeat timings retained explicitly. This round teacher predictions and confidence exactly equal frozen four-mode baselines. Instrumented 1-repeat model timings are not used as performance evidence.',
        'retained_structure_scope': 'No new candidate end-to-end RMSD measured. Original PPU TF32 vs native H100 TF32 interval remains [0.086655040494,0.218147146445] Angstrom, unresolved at 0.1 Angstrom threshold.',
        'remaining': ['Do not mistake native Q drift for PPU rounding error or imitate the drift.',
                      'Optimize actual backward dataflow and conversion/packing cost in dQ and dK/dV; use generic algorithms and measure whole operator.',
                      'No corrected H100 reference substituted for native project acceptance.',
                      'Nine-model multi-input candidate acceptance and full-model training remain outstanding.'],
        'source_evidence_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for side in ('h100', 'ppu') for path in (ROOT / side).rglob('*.json')}}
    (ROOT / 'complete_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print('QUERY_ONCE_COMPARISON_COMPLETE', len(timing_rows), len(accuracy_rows), len(candidate_rows), len(live_rows))


if __name__ == '__main__':
    main()
