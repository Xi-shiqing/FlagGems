"""Reproducible four-mode reports from scalar/hash evidence only."""
import csv
from collections import Counter
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
PRECISIONS = ('fp32', 'tf32')
CONFIDENCES = ('plddt', 'gpde', 'ptm', 'iptm', 'ranking_score')


def read(path):
    return json.loads(path.read_text())


def save_csv(name, rows):
    with (ROOT / name).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows, fields):
    def cell(value):
        if isinstance(value, float):
            return f'{value:.3e}' if 0 < abs(value) < .001 else f'{value:.4f}'
        return str(value)
    return '\n'.join(['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join('---' for _ in fields) + ' |'] +
                     ['| ' + ' | '.join(cell(row[f]) for f in fields) + ' |' for row in rows])


def interval(ppu, h100, precision, anchors, extra_anchors=()):
    lowers, uppers, exact = [], [], []
    for p in PRECISIONS:
        anchor = anchors[(ppu['case'], ppu['model'], 'ppu_' + p)]
        assert anchor['prediction_sha256'] == ppu['anchor_prediction_sha256'][p]
        for h in PRECISIONS:
            distance = anchor['vs_h100_' + h]['structure']['kabsch_rmsd_A']
            dp = ppu['rmsd_to_own_frozen_anchors_A'][precision][p]
            dh = h100['rmsd_to_own_frozen_anchors_A'][precision][h]
            lowers.append(max(0, distance - dp - dh, dp - distance - dh, dh - distance - dp))
            uppers.append(distance + dp + dh)
            if dp == 0 and dh == 0:
                exact.append(distance)
    for anchor in extra_anchors:
        assert anchor['input_audit'] == 'PASS'
        for h in PRECISIONS:
            distance = anchor['vs_h100_' + h]['structure']['kabsch_rmsd_A']
            dp = anchor['rmsd_to_anchor_A'][precision]
            dh = h100['rmsd_to_own_frozen_anchors_A'][precision][h]
            lowers.append(max(0, distance - dp - dh, dp - distance - dh, dh - distance - dp))
            uppers.append(distance + dp + dh)
    lower, upper = max(lowers), min(uppers)
    assert lower <= upper + 1e-8
    status = 'PASS' if upper <= .1 else 'FAIL' if lower > .1 else 'UNRESOLVED'
    return {'lower_A': lower, 'upper_A': upper, 'status': status, 'exact_A': exact[0] if exact else None,
            'anchor_interval_count': len(lowers),
            'method': 'Intersection of triangle-inequality intervals using measured current-to-own-anchor RMSDs, including available frozen mixed-policy anchors; native cross-device anchors are immutable.'}


def main():
    report = {'status': 'MEASUREMENTS_COMPLETE', 'operator': [], 'matrix': [], 'training': []}
    pbench, hbench = [read(ROOT / s / 'combined_bench.json') for s in ('ppu', 'h100')]
    operator_rows, operator_accuracy = [], []
    for pc, hc in zip(pbench['cases'], hbench['cases']):
        assert pc['shape'] == hc['shape'] and pc['full_input_sha256'] == hc['full_input_sha256']
        pm, hm = pc['modes'], hc['modes']
        def timing(modes, mode, kind):
            values = modes[mode]['timing_median'][kind]
            return values['event_ms'] if values else 'N/A'
        for kind in ('inference', 'training'):
            row = {'S': pc['shape'][2], 'scope': kind, 'H100_FP32_ms': timing(hm, 'fp32', kind),
                'H100_TF32_ms': timing(hm, 'tf32', kind), 'PPU_FP32_ms': timing(pm, 'fp32', kind),
                'PPU_TF32_ms': timing(pm, 'tf32', kind), 'PPU_combined_TF32_ms': timing(pm, 'combined_64m', kind)}
            row['new_vs_old_TF32_speedup'] = row['PPU_TF32_ms'] / row['PPU_combined_TF32_ms']
            operator_rows.append(row)
        memory = {mode: pm[mode]['accuracy_call_peak_extra_bytes'] / 1024**2 for mode in ('tf32', 'combined_64m', 'combined_128m', 'combined_256m')}
        report['operator'].append({'shape': pc['shape'], 'memory_peak_extra_MiB': memory})
        for label, modes, mode in [('H100_FP32', hm, 'fp32'), ('H100_TF32', hm, 'tf32'),
                                   ('PPU_FP32', pm, 'fp32'), ('PPU_TF32', pm, 'tf32'), ('PPU_combined_TF32', pm, 'combined_64m')]:
            record = modes[mode]
            operator_accuracy.append({'S': pc['shape'][2], 'mode': label, 'output_relative_l2': record['output']['relative_l2'],
                'max_gradient_relative_l2': max((x['gradient']['relative_l2'] for x in record['parameters'].values()), default='N/A'),
                'max_realized_update_relative_l2': max((x['realized_fp32_update']['relative_l2'] for x in record['parameters'].values()), default='N/A'),
                'training_status': record['training_status']})
    save_csv('operator_timings.csv', operator_rows)
    save_csv('operator_accuracy.csv', operator_accuracy)

    frozen = read(ROOT / 'frozen_cross_device_anchors.json')['rows']
    anchors = {(r['case'], r['model'], r['mode']): r for r in frozen}
    extra_file = ROOT / 'ppu/extra_anchor_distances.json'
    extras = read(extra_file)['cases'] if extra_file.exists() else []
    original_pmat = read(ROOT / 'ppu/derived_matrix.json')
    serial_path = ROOT / 'ppu/derived_matrix_serial.json'
    pmat = read(serial_path) if serial_path.exists() else original_pmat
    hmat = read(ROOT / 'h100/derived_matrix.json')
    if serial_path.exists():
        originals = {(r['case'], r['model']): r for r in original_pmat['cases']}
        for current in pmat['cases']:
            old = originals[(current['case'], current['model'])]
            for mode in current['summary']:
                for key in ('prediction_digest', 'five_confidence_metrics'):
                    assert current['summary'][mode][key] == old['summary'][mode][key]
    report['matrix_timing_scope'] = 'PPU serial, no concurrent training by this experiment' if serial_path.exists() else 'Parallel screening: contaminated timing, do not claim speed gains'
    if not serial_path.exists():
        report['status'] = 'PENDING_SERIAL_TIMING'
    native_anchors = read(ROOT / 'native_anchor_hashes.json')
    hm = {(r['case'], r['model']): r for r in hmat['cases']}
    timing_rows, confidence_rows, rmsd_rows = [], [], []
    for p in pmat['cases']:
        h = hm[(p['case'], p['model'])]
        for key in ('checkpoint_sha256', 'N_token', 'atom_count', 'features', 'atom_identity_sha256', 'canonical_input_sha256', 'asset_hashes'):
            assert p[key] == h[key], (p['case'], p['model'], key)
        for key in ('initialization_semantic_sha256', 'prediction_semantic_sha256'):
            assert p['rng'][key] == h['rng'][key], (p['case'], p['model'], key)
        assert h['anchor_prediction_sha256'] == native_anchors[p['case'] + '/' + p['model']]
        assert p['candidate_rmsd_vs_current_tf32_A'] == 0
        ps, hs = p['summary'], h['summary']
        assert ps['combined_tf32']['five_confidence_metrics'] == ps['tf32']['five_confidence_metrics']
        row = {'input': p['case'], 'tokens': p['N_token'], 'model': p['model'],
            'H100_FP32_s': hs['fp32']['median_seconds'], 'H100_TF32_s': hs['tf32']['median_seconds'],
            'PPU_FP32_s': ps['fp32']['median_seconds'], 'PPU_TF32_s': ps['tf32']['median_seconds'],
            'PPU_combined_TF32_s': ps['combined_tf32']['median_seconds']}
        row['new_vs_old_TF32_change_pct'] = 100 * (row['PPU_combined_TF32_s'] / row['PPU_TF32_s'] - 1)
        row['PPU_TF32_vs_H100_TF32_time_ratio'] = row['PPU_TF32_s'] / row['H100_TF32_s']
        row['timing_control'] = 'serial' if serial_path.exists() else 'DISCARDED_same_GPU_competition'
        timing_rows.append(row)
        gates = {}
        for precision in PRECISIONS:
            metrics, passing = {}, True
            for metric in CONFIDENCES:
                pv, hv = ps[precision]['five_confidence_metrics'][metric], hs[precision]['five_confidence_metrics'][metric]
                difference = abs(pv - hv)
                relative = 100 * difference / abs(hv) if hv != 0 else (0 if difference == 0 else None)
                valid = (difference == 0) if hv == 0 else relative <= 5
                passing = passing and valid
                metrics[metric] = {'absolute_difference': difference, 'relative_difference_pct': relative, 'pass_le_5pct': valid}
            structure = interval(p, h, precision, anchors, [e for e in extras if e['case'] == p['case'] and e['model'] == p['model']])
            gate = 'FAIL' if not passing or structure['status'] == 'FAIL' else structure['status']
            gates[precision] = {'status': gate, 'five_confidence': metrics, 'rmsd': structure}
        for metric in CONFIDENCES:
            confidence_rows.append({'input': p['case'], 'model': p['model'], 'metric': metric,
                'H100_FP32': hs['fp32']['five_confidence_metrics'][metric], 'H100_TF32': hs['tf32']['five_confidence_metrics'][metric],
                'PPU_FP32': ps['fp32']['five_confidence_metrics'][metric], 'PPU_TF32': ps['tf32']['five_confidence_metrics'][metric],
                'PPU_combined_TF32': ps['combined_tf32']['five_confidence_metrics'][metric],
                'FP32_delta_pct': gates['fp32']['five_confidence'][metric]['relative_difference_pct'],
                'TF32_delta_pct': gates['tf32']['five_confidence'][metric]['relative_difference_pct']})
        rmsd_rows.append({'input': p['case'], 'model': p['model'],
            'PPU_FP32_vs_H100_FP32_lower_A': gates['fp32']['rmsd']['lower_A'], 'PPU_FP32_vs_H100_FP32_upper_A': gates['fp32']['rmsd']['upper_A'],
            'FP32_gate': gates['fp32']['status'], 'PPU_TF32_vs_H100_TF32_lower_A': gates['tf32']['rmsd']['lower_A'],
            'PPU_TF32_vs_H100_TF32_upper_A': gates['tf32']['rmsd']['upper_A'], 'TF32_gate': gates['tf32']['status'],
            'candidate_vs_current_PPU_TF32_RMSD_A': 0,
            'PPU_TF32_vs_local_FP32_RMSD_A': p.get('local_tf32_vs_fp32_rmsd_A', 'N/A'),
            'H100_TF32_vs_local_FP32_RMSD_A': h.get('local_tf32_vs_fp32_rmsd_A', 'N/A')})
        report['matrix'].append({'case': p['case'], 'model': p['model'], 'input_audit': 'PASS', 'gates': gates, 'timing': row})
    save_csv('nine_model_timings.csv', timing_rows)
    save_csv('nine_model_confidence.csv', confidence_rows)
    save_csv('nine_model_rmsd.csv', rmsd_rows)
    report['matrix_gate_counts'] = {precision: dict(Counter(r['gates'][precision]['status'] for r in report['matrix'])) for precision in PRECISIONS}
    changes = [r['new_vs_old_TF32_change_pct'] for r in timing_rows]
    report['matrix_candidate_effect'] = {'case_count': len(changes), 'median_change_pct': statistics.median(changes),
        'range_change_pct': [min(changes), max(changes)], 'faster_cases': sum(v < 0 for v in changes),
        'accuracy_vs_current_ppu_tf32': 'Exact coordinates and five confidences for all 18 cases',
        'timing_limitation': '1 warmup and 3 paired repeats per mode; screening only, small percent differences do not establish a robust gain.'}

    ptrain, htrain = [read(ROOT / s / 'derived_training.json') for s in ('ppu', 'h100')]
    assert ptrain['rng_program_sha256'] == htrain['rng_program_sha256']
    training_rows, training_accuracy = [], []
    for pc, hc in zip(ptrain['cases'], htrain['cases']):
        assert pc['crop'] == hc['crop']
        pm, hm = pc['modes'], hc['modes']
        for precision in PRECISIONS:
            for key in ('batch_leaf_hashes', 'batch_sha256', 'rng_semantic_sha256', 'initial_parameter_sha256'):
                assert pm[precision][key] == hm[precision][key], (pc['crop'], precision, key)
            assert sorted(pm[precision]['input_hashes'].values()) == sorted(hm[precision]['input_hashes'].values())
        row = {'tokens': pc['crop'], 'H100_FP32_ms': hm['fp32']['formal_event_median_ms'], 'H100_TF32_ms': hm['tf32']['formal_event_median_ms'],
               'PPU_FP32_ms': pm['fp32']['formal_event_median_ms'], 'PPU_TF32_ms': pm['tf32']['formal_event_median_ms'],
               'PPU_combined_TF32_ms': pm['combined_tf32']['formal_event_median_ms']}
        row['new_vs_old_TF32_change_pct'] = 100 * (row['PPU_combined_TF32_ms'] / row['PPU_TF32_ms'] - 1)
        training_rows.append(row)
        for step in range(4):
            training_accuracy.append({'tokens': pc['crop'], 'step': step, 'H100_FP32_loss': hm['fp32']['losses'][step]['loss'],
                'H100_TF32_loss': hm['tf32']['losses'][step]['loss'], 'PPU_FP32_loss': pm['fp32']['losses'][step]['loss'],
                'PPU_TF32_loss': pm['tf32']['losses'][step]['loss'], 'PPU_combined_TF32_loss': pm['combined_tf32']['losses'][step]['loss']})
        report['training'].append({'crop': pc['crop'], 'input_and_rng_audit': 'PASS', 'timing': row,
            'candidate_vs_local_tf32': pm['combined_tf32']['vs_local_tf32'],
            'H100_tf32_vs_local_fp32': hm['tf32']['vs_local_fp32'], 'PPU_tf32_vs_local_fp32': pm['tf32']['vs_local_fp32'],
            'cross_device_full_gradient_l2': 'Not directly computed; complete gradients remain on their original servers. Scalar norm/hash audits and cross-device loss results are exported.'})
    save_csv('full_training_timings_parallel_screening.csv', training_rows)
    save_csv('full_training_losses.csv', training_accuracy)
    confirmed = ROOT / 'ppu/training_confirm/245'
    if all((confirmed / mode / 'training_report.json').exists() for mode in ('fp32', 'tf32', 'combined_tf32')):
        cases = ptrain['cases'][1]['modes']
        confirmation = {mode: read(confirmed / mode / 'training_report.json') for mode in ('fp32', 'tf32', 'combined_tf32')}
        for mode, value in confirmation.items():
            assert value['status'] == 'pass'
            assert value['batch_sha256'] == cases[mode]['batch_sha256']
            assert all(x['loss'] == x['loss'] for x in value['losses'])
            value['loss_repeat_delta'] = [x['loss'] - y['loss'] for x, y in zip(value['losses'], cases[mode]['losses'])]
        native = htrain['cases'][1]['modes']
        training_rows = [{'tokens': 245, 'H100_FP32_ms': native['fp32']['formal_event_median_ms'],
            'H100_TF32_ms': native['tf32']['formal_event_median_ms'],
            'PPU_FP32_ms': confirmation['fp32']['formal_event_median_ms'],
            'PPU_TF32_ms': confirmation['tf32']['formal_event_median_ms'],
            'PPU_combined_TF32_ms': confirmation['combined_tf32']['formal_event_median_ms'],
            'new_vs_old_TF32_change_pct': 100 * (confirmation['combined_tf32']['formal_event_median_ms'] / confirmation['tf32']['formal_event_median_ms'] - 1)}]
        report['training_confirmation'] = {'scope': 'Warm-cache PPU performance rerun after inference workers finished; 4 steps, no tensor audit overhead, last 3 timed. H100 matched full training baselines. Small sample performance evidence.', 'rows': training_rows,
            'loss_repeat_deltas': {mode: value['loss_repeat_delta'] for mode, value in confirmation.items()}}
    final_confirmation = read(confirmed / 'backward_only_tf32/training_report.json')
    final_entry = read(confirmed / 'backward_only_tf32/actual_training_entry.json')
    identity = read(ROOT / 'ppu/backward_only_route_identity.json')
    controls = read(ROOT / 'ppu/backward_only_controls.json')
    assert final_confirmation['status'] == 'pass'
    assert final_confirmation['batch_leaf_hashes'] == confirmation['tf32']['batch_leaf_hashes']
    assert final_entry['matmul_allow_tf32'] and final_entry['candidate_physical_backward_calls'] > 0
    assert identity['status'] == controls['status'] == 'PASS'
    row = training_rows[0]
    row['PPU_final_backward_only_TF32_ms'] = final_confirmation['formal_event_median_ms']
    row['final_vs_current_TF32_change_pct'] = 100 * (row['PPU_final_backward_only_TF32_ms'] / row['PPU_TF32_ms'] - 1)
    row['final_PPU_TF32_vs_H100_TF32_time_ratio'] = row['PPU_final_backward_only_TF32_ms'] / row['H100_TF32_ms']
    report['selected_integration'] = {
        'candidate': 'backward_only', 'scope': 'Experimental real-training integration; original public forward and original IEEE backward retained.',
        'workspace_budget_bytes': 64 * 1024**2, 'route_identity': identity, 'controls': controls,
        'timing': row, 'inference': 'Original public forward function preserved by identity; no new inference timing is fabricated.',
        'rejected_combined_forward_reason': 'Serial 18-case screening found no stable inference gain and several regressions.',
        'measured_step_timings': final_confirmation['timings'],
        'training_dispatch_scope': 'FlagGems TriAttention explicitly enabled and observed; global enable_flaggems=false, triangle_multiplicative=torch. No claim that all training MM/addmm/BMM dispatch to FlagGems.',
        'training_sample_limit': '245-token Protenix-v2 only, 3 timed steps after 1 warmup; about 3% measured saving is preliminary, not a demonstrated steady-state guarantee.'}
    save_csv('full_training_timings.csv', training_rows)
    save_csv('final_full_training_losses.csv', [
        {'tokens': 245, 'step': step, 'H100_FP32_loss': native['fp32']['losses'][step]['loss'],
         'H100_TF32_loss': native['tf32']['losses'][step]['loss'],
         'PPU_FP32_loss': confirmation['fp32']['losses'][step]['loss'],
         'PPU_TF32_loss': confirmation['tf32']['losses'][step]['loss'],
         'PPU_final_backward_only_TF32_loss': final_confirmation['losses'][step]['loss']}
        for step in range(4)])
    if all((ROOT / s / 'derived_nonzero_update.json').exists() for s in ('ppu', 'h100')):
        nonzero = {s: read(ROOT / s / 'derived_nonzero_update.json') for s in ('ppu', 'h100')}
        assert nonzero['ppu']['rng_program_sha256'] == nonzero['h100']['rng_program_sha256']
        report['nonzero_update'] = []
        accuracy_rows = []
        for pc, hc in zip(nonzero['ppu']['cases'], nonzero['h100']['cases']):
            for mode in ('fp32', 'tf32'):
                for key in ('batch_leaf_hashes', 'rng_semantic_sha256', 'initial_parameter_sha256'):
                    assert pc['modes'][mode][key] == hc['modes'][mode][key]
            for side, case in (('ppu', pc), ('h100', hc)):
                for mode, value in case['modes'].items():
                    audit = value['parameter_audit']
                    assert audit['audit_step'] == 1 and audit['nonzero_update_assertion'] == 'PASS'
                    assert all(lr > 0 for lr in audit['optimizer_learning_rates_before_step'])
                    error = value.get('vs_local_tf32', value.get('vs_local_fp32'))
                    accuracy_rows.append({'tokens': case['crop'], 'device': side, 'precision': mode,
                        'reference': 'same-device current TF32' if mode == 'combined_tf32' else 'same-device strict FP32',
                        'gradient_relative_l2': error['gradients']['global_relative_l2'] if error else 0,
                        'update_relative_l2': error['updates']['global_relative_l2'] if error else 0,
                        'update_max_abs': error['updates']['global_max_abs'] if error else 0,
                        'audit_step': 1, 'learning_rate': audit['optimizer_learning_rates_before_step'][0]})
            report['nonzero_update'].append({'crop': pc['crop'], 'candidate_vs_local_tf32': pc['modes']['combined_tf32']['vs_local_tf32'],
                'PPU_tf32_vs_local_fp32': pc['modes']['tf32']['vs_local_fp32'], 'H100_tf32_vs_local_fp32': hc['modes']['tf32']['vs_local_fp32'],
                'input_rng_initial_parameter_audit': 'PASS'})
        report['selected_integration']['nonzero_update'] = []
        for crop in (131, 245):
            path = ROOT / f'ppu/nonzero_update/{crop}/backward_only_tf32'
            audit = read(path / 'actual_parameter_audit.json')
            entry = read(path / 'actual_training_entry.json')
            assert audit['audit_step'] == 1 and audit['nonzero_update_assertion'] == 'PASS'
            assert audit['parameter_count'] == 4174 and audit['gradient_parameter_count'] == 4170
            assert entry['matmul_allow_tf32'] and entry['candidate_physical_backward_calls'] > 0
            assert all(value['finite'] for value in audit['comparisons'].values())
            error = audit['comparisons']
            accuracy_rows.append({'tokens': crop, 'device': 'ppu', 'precision': 'final_backward_only_tf32',
                'reference': 'same-device current TF32',
                'gradient_relative_l2': error['gradients']['global_relative_l2'],
                'update_relative_l2': error['updates']['global_relative_l2'],
                'update_max_abs': error['updates']['global_max_abs'],
                'audit_step': 1, 'learning_rate': audit['optimizer_learning_rates_before_step'][0]})
            report['selected_integration']['nonzero_update'].append({'crop': crop, 'audit': audit, 'actual_entry': entry})
        save_csv('full_parameter_nonzero_update_accuracy.csv', accuracy_rows)
    report['zero_learning_rate_step0_update'] = 'DISCARDED as update-accuracy evidence. Only audit_step=1 with positive learning rate is valid.'
    (ROOT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# TF32 通用反向缓存与四模式对照（2026-10-03）', '',
        '最终保留的实验接入方案为 backward_only：只在 PPU TF32 反向使用通用分块缓存，保留原有公开前向及 IEEE 反向。固定 64 MiB 临时缓存按实际字节数分块，直接写入最终梯度；没有新增模型或 shape 白名单。已经通过真实训练入口验证，尚未修改正式算子包的默认入口。', '',
        f"245-token 真实训练单独复测：当前 PPU TF32 {row['PPU_TF32_ms']/1000:.4f} 秒，最终方案 {row['PPU_final_backward_only_TF32_ms']/1000:.4f} 秒，耗时减少 {-row['final_vs_current_TF32_change_pct']:.2f}%。只有 3 个正式步骤，不能据此保证长期稳定收益。", '',
        '合并前后向的 combined 候选仅作为实验记录保留：18 组串行推理耗时变化中位数为 +0.08%，没有稳定推理收益，故其前向改动不纳入最终方案。以下 combined 列均为未采用候选。最终方案前向函数身份与原实现相同，不复制基线耗时冒充新测量。', '',
        '算子耗时单位 ms。输入是 CPU 统一随机数生成的合成张量（B=1、N=S、H=4、D=32），不是实测模型激活。训练包含前向、反向、一次 SGD 更新；H100 原生 CUE 0.8 的严格 FP32 反向不支持，明确标为 N/A。实际完整模型训练使用官方 Protenix 训练 TriAttention（支持 IEEE），与推理 CUE 入口分别记录。', '',
        table(operator_rows, list(operator_rows[0])), '',
        '真实训练：Protenix-v2，48 Pairformer + 4 MSA blocks，131/245 tokens，4 个训练步骤，统计后 3 步。并行首轮耗时仅留档，不作为加速结论；245 tokens 在自有推理工作结束后单独复测。训练沿用既有入口，明确接入并计数的 FlagGems 算子是 TriAttention；全局 enable_flaggems=false，triangle_multiplicative=torch，不能把这组训练称为所有算子都走 FlagGems。两服务器统一随机数程序 SHA、增强后全部输入、初始参数和有序随机事件均一致。', '',
        table(training_rows, list(training_rows[0])), '',
        '九模型真实推理：G01=693 tokens，G02=245 tokens；1 warmup + 3 次交错重复，cycle/step/sample=1。PPU 正式性能结果使用无自有训练并行的串行复测，预测和置信度必须与首轮完全相同。这里只代表这个测试协议，不能作为完整扩散采样的性能结论。', '',
        table(timing_rows, list(timing_rows[0])), '',
        'RMSD 门槛为 0.1 Å，五项置信度逐项相对误差门槛为 5%。跨设备坐标不导出，本轮使用已冻结的直接跨设备 RMSD 和各服务器实测的新旧预测漂移，计算严格区间；另加入 22 个已有混合精度锚点收紧区间，仍跨门槛时必须记为 UNRESOLVED。候选与当前 PPU TF32 的坐标和五项置信度全部完全相同。', '',
        table(rmsd_rows, list(rmsd_rows[0])), '',
        table(accuracy_rows, list(accuracy_rows[0])), '',
        '五项置信度的完整四模式数值见 nine_model_confidence.csv；算子输出/梯度/更新误差见 operator_accuracy.csv。第 0 步学习率为零，其零更新不能作为精度证据；已另外重跑第 1 步的非零 Adam 更新（学习率 0.00018），包括最终 backward_only，覆盖全部 4170 个有梯度参数及 4174 个参数更新。最终方案相对当前 PPU TF32 的梯度相对 L2 为 3.90e-6～4.28e-6，Adam 更新相对 L2 为 0.108%～0.342%，最大绝对更新差约 2.6e-4～2.9e-4；不能把它描述为更新逐位一致或已有业务阈值全部通过。完整训练的跨设备全参数梯度 L2 尚未直接计算；真实训练只覆盖 Protenix-v2 的两个裁剪长度，并非九模型训练验收。', '',
        f"筛选门槛：FP32 {report['matrix_gate_counts']['fp32']}；TF32 {report['matrix_gate_counts']['tf32']}。TF32 未整体过关；未决 RMSD 保留未决标记，未作为通过处理。", '',
        'PPU 首轮两组推理和训练都被 env_ppu.sh 重置到 GPU 7，产生同卡竞争，首轮并行耗时作废；GPU 选择已放在环境脚本之后，正式串行推理与训练复测使用 GPU 7、无本实验的并发任务。H100 最初训练记录因加载旧随机数程序而作废，保存于 training_failed_rng；精度开关被模型构造函数重置而未进入训练的运行保存于 training_failed_control。有效训练显式设置 configs.enable_tf32，并设置官方训练 TriAttention 的 deterministic 参数，记录真实调用数。训练性能复测的损失与数值审计运行存在少量重复差异，具体差值记录于 summary.json；不声称训练逐位确定。', '',
        '所有原始模型参数、激活、坐标和梯度快照保留在原服务器；本目录只含自有代码、标量和哈希。3 次重复的微小耗时差异只作筛选证据，不宣称稳定的端到端提升。']
    if not serial_path.exists():
        lines.insert(2, '中间草稿：九模型串行计时尚未完成。下方标有 DISCARDED 的推理速度来自同 GPU 并发首轮，不能用于性能结论。')
    (ROOT / 'report.md').write_text('\n'.join(lines) + '\n')
    print('CLOSURE_SUMMARY', report['matrix_gate_counts'], report['matrix_candidate_effect'], training_rows)


if __name__ == '__main__':
    main()
