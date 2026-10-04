"""Complete four-mode tables plus explicitly labeled native-reference diagnostics."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
METRICS = ('plddt', 'gpde', 'ptm', 'iptm', 'ranking_score')
HMODES = ('fp32', 'tf32', 'tf32_triangle_ieee', 'tf32_triangle_corrected')


def read(path):
    return json.loads(path.read_text())


def save_csv(name, rows):
    with (ROOT / name).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows):
    fields = list(rows[0])
    def cell(x):
        if isinstance(x, float):
            return f'{x:.3e}' if 0 < abs(x) < .001 else f'{x:.4f}'
        return str(x)
    return '\n'.join(['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join('---' for _ in fields) + ' |'] +
        ['| ' + ' | '.join(cell(row[key]) for key in fields) + ' |' for row in rows])


def distance_interval(p, h, pmode, hmode, anchors):
    lowers, uppers = [], []
    for pa in ('fp32', 'tf32'):
        anchor = anchors[(p['case'], p['model'], 'ppu_' + pa)]
        assert anchor['prediction_sha256'] == p['anchor_prediction_sha256'][pa]
        for ha in ('fp32', 'tf32'):
            distance = anchor['vs_h100_' + ha]['structure']['kabsch_rmsd_A']
            dp = p['rmsd_to_own_frozen_anchors_A'][pmode][pa]
            dh = h['rmsd_to_own_frozen_anchors_A'][hmode][ha]
            lowers.append(max(0, distance-dp-dh, dp-distance-dh, dh-distance-dp))
            uppers.append(distance+dp+dh)
    lower, upper = max(lowers), min(uppers)
    assert lower <= upper + 1e-8
    return {'lower_A': lower, 'upper_A': upper,
        'status': 'PASS' if upper <= .1 else 'FAIL' if lower > .1 else 'UNRESOLVED'}


def confidence_gate(p, h):
    rows = {}
    for metric in METRICS:
        pv, hv = p[metric], h[metric]
        delta = abs(pv-hv)
        relative = 100*delta/abs(hv) if hv else (0 if not delta else None)
        rows[metric] = {'relative_difference_pct': relative, 'pass': relative is not None and relative <= 5}
    return all(row['pass'] for row in rows.values()), rows


def main():
    sides = {side: read(ROOT / side / 'derived.json') for side in ('ppu', 'h100')}
    assert all(x['status'] == 'PASS' and len(x['cases']) == 18 for x in sides.values())
    adapter = read(ROOT / 'h100/adapter_validation_fixed.json')
    assert adapter['status'] == 'PASS' and len(adapter['cases']) == 9
    anchors = {(r['case'], r['model'], r['mode']): r for r in read(ROOT / 'frozen_cross_device_anchors.json')['rows']}
    hcases = {(r['case'], r['model']): r for r in sides['h100']['cases']}
    timings, confidences, drifts, structures, case_reports = [], [], [], [], []
    native_counts, diagnostic_counts = {'fp32': Counter(), 'tf32': Counter()}, {m: Counter() for m in HMODES[2:]}
    for p in sides['ppu']['cases']:
        h = hcases[(p['case'], p['model'])]
        for key in ('checkpoint_sha256', 'N_token', 'atom_count', 'features', 'atom_identity_sha256', 'canonical_input_sha256', 'asset_hashes'):
            assert p[key] == h[key], (p['case'], p['model'], key)
        for key in ('initialization_semantic_sha256', 'prediction_semantic_sha256'):
            assert p['rng'][key] == h['rng'][key]
        assert p['frozen_baseline_predictions_retained'] and h['frozen_baseline_predictions_retained']
        assert all(h['triangle_calls_first_warmup'][m] > 0 for m in HMODES)
        ps, hs = p['summary'], h['summary']
        base = {'input': p['case'], 'tokens': p['N_token'], 'model': p['model']}
        timing = base | {'H100_native_FP32_s': hs['fp32']['median_seconds'], 'H100_native_TF32_s': hs['tf32']['median_seconds'],
            'PPU_FlagGems_FP32_s': ps['fp32']['median_seconds'], 'PPU_FlagGems_TF32_s': ps['tf32']['median_seconds'],
            'H100_mixed_triangle_IEEE_s': hs[HMODES[2]]['median_seconds'], 'H100_custom_triangle_RNE_s': hs[HMODES[3]]['median_seconds'],
            'PPU_TF32_vs_native_H100_TF32_time_ratio': ps['tf32']['median_seconds']/hs['tf32']['median_seconds'],
            'PPU_gpu': p['physical_gpu'], 'H100_gpu': h['physical_gpu']}
        timings.append(timing)
        for metric in METRICS:
            confidences.append(base | {'metric': metric,
                'H100_native_FP32': hs['fp32']['five_confidence_metrics'][metric], 'H100_native_TF32': hs['tf32']['five_confidence_metrics'][metric],
                'PPU_FlagGems_FP32': ps['fp32']['five_confidence_metrics'][metric], 'PPU_FlagGems_TF32': ps['tf32']['five_confidence_metrics'][metric],
                'H100_mixed_triangle_IEEE': hs[HMODES[2]]['five_confidence_metrics'][metric],
                'H100_custom_triangle_RNE': hs[HMODES[3]]['five_confidence_metrics'][metric]})
        native_drift = h['rmsd_to_current_local_fp32_A']['tf32']
        corrected_drift = h['rmsd_to_current_local_fp32_A'][HMODES[3]]
        drifts.append(base | {'H100_native_FP32_vs_own_FP32_A': 0,
            'H100_native_TF32_vs_own_FP32_A': native_drift, 'PPU_FP32_vs_own_FP32_A': 0,
            'PPU_TF32_vs_own_FP32_A': p['rmsd_to_current_local_fp32_A']['tf32'],
            'H100_mixed_triangle_IEEE_vs_own_FP32_A': h['rmsd_to_current_local_fp32_A'][HMODES[2]],
            'H100_custom_triangle_RNE_vs_own_FP32_A': corrected_drift,
            'H100_custom_triangle_RNE_vs_native_TF32_A': h['rmsd_to_current_local_tf32_A'][HMODES[3]],
            'H100_drift_reduction_pct': 100*(1-corrected_drift/native_drift) if native_drift else None})
        gates = {}
        for pmode, hmode, label in [('fp32', 'fp32', 'native_fp32'), ('tf32', 'tf32', 'native_tf32'),
                                  ('tf32', HMODES[2], 'mixed_triangle_ieee'), ('tf32', HMODES[3], 'custom_triangle_rne')]:
            distance = distance_interval(p, h, pmode, hmode, anchors)
            passing, metrics = confidence_gate(ps[pmode]['five_confidence_metrics'], hs[hmode]['five_confidence_metrics'])
            gate = 'FAIL' if not passing or distance['status'] == 'FAIL' else distance['status']
            gates[label] = {'status': gate, 'rmsd': distance, 'five_confidence': metrics}
            if label.startswith('native_'):
                native_counts[pmode][gate] += 1
            else:
                diagnostic_counts[hmode][gate] += 1
        structures.append(base | {field: value for label, g in gates.items() for field, value in
            [(label+'_lower_A', g['rmsd']['lower_A']), (label+'_upper_A', g['rmsd']['upper_A']), (label+'_gate', g['status'])]})
        case_reports.append(base | {'input_and_rng_audit': 'PASS', 'frozen_four_mode_predictions_retained': True,
            'timing': timing, 'same_device_structure': drifts[-1], 'cross_device_gates': gates})
    save_csv('four_mode_and_diagnostic_timings.csv', timings)
    save_csv('four_mode_and_diagnostic_confidence.csv', confidences)
    save_csv('four_mode_and_diagnostic_structure_drift.csv', drifts)
    save_csv('cross_device_structure_intervals.csv', structures)
    head = read(ROOT / 'h100/derived_confidence.json')
    assert head['status'] == 'PASS' and len(head['cases']) == 6
    pcases = {(r['case'], r['model']): r for r in sides['ppu']['cases']}
    reports = {(r['input'], r['model']): r for r in case_reports}
    head_timings, head_confidences, head_cases = [], [], []
    head_counts = Counter()
    for row in head['cases']:
        key = row['case'], row['model']
        p, h, previous = pcases[key], hcases[key], reports[key]
        assert row['input_and_rng_audit'] == 'PASS' and row['coordinate_hash_unchanged']
        assert row['previous_corrected'] == h['summary'][HMODES[3]]
        assert row['precision_audit']['status'] == 'PASS'
        current = row['summary']
        gate_pass, metrics = confidence_gate(p['summary']['tf32']['five_confidence_metrics'], current['five_confidence_metrics'])
        interval = previous['cross_device_gates']['custom_triangle_rne']['rmsd']
        gate = 'FAIL' if not gate_pass or interval['status'] == 'FAIL' else interval['status']
        head_counts[gate] += 1
        head_timings.append(previous['timing'] | {'H100_custom_RNE_confidence_matmul_IEEE_s': current['median_seconds']})
        for metric in METRICS:
            base_row = next(r for r in confidences if (r['input'], r['model'], r['metric']) == (*key, metric))
            head_confidences.append(base_row | {'H100_custom_RNE_confidence_matmul_IEEE': current['five_confidence_metrics'][metric],
                'PPU_TF32_relative_error_before_pct': previous['cross_device_gates']['custom_triangle_rne']['five_confidence'][metric]['relative_difference_pct'],
                'PPU_TF32_relative_error_after_pct': metrics[metric]['relative_difference_pct']})
        head_cases.append({'input': key[0], 'model': key[1], 'coordinate_hash_unchanged': True,
            'input_and_rng_audit': 'PASS', 'before_gate': previous['cross_device_gates']['custom_triangle_rne']['status'],
            'after_gate': gate, 'rmsd_interval_unchanged': interval, 'five_confidence': metrics,
            'precision_audit': row['precision_audit']})
    save_csv('confidence_scope_timings.csv', head_timings)
    save_csv('confidence_scope_metrics.csv', head_confidences)
    summary = {'status': 'NATIVE_REFERENCE_MODEL_ABLATION_COMPLETE', 'case_count': len(case_reports),
        'native_gate_counts': {m: dict(c) for m, c in native_counts.items()},
        'diagnostic_gate_counts': {m: dict(c) for m, c in diagnostic_counts.items()},
        'median_native_structure_drift_A': statistics.median(r['H100_native_TF32_vs_own_FP32_A'] for r in drifts),
        'median_corrected_structure_drift_A': statistics.median(r['H100_custom_triangle_RNE_vs_own_FP32_A'] for r in drifts),
        'median_drift_reduction_pct': statistics.median(r['H100_drift_reduction_pct'] for r in drifts),
        'same_device_structure_gate_counts': {
            'H100_native_TF32_vs_own_FP32': sum(r['H100_native_TF32_vs_own_FP32_A']<=.1 for r in drifts),
            'H100_custom_triangle_RNE_vs_own_FP32': sum(r['H100_custom_triangle_RNE_vs_own_FP32_A']<=.1 for r in drifts),
            'PPU_FlagGems_TF32_vs_own_FP32': sum(r['PPU_TF32_vs_own_FP32_A']<=.1 for r in drifts)},
        'adapter_validation': adapter, 'scope': read(ROOT / 'scope.json'), 'cases': case_reports,
        'confidence_scope_ablation': {'tested_case_count': 6, 'gate_counts': dict(head_counts), 'cases': head_cases,
            'scope': 'H100 custom RNE TriAttention with IEEE matmul only during the ConfidenceHead class. TriAttention remains explicit TF32. No PPU policy or 18-case acceptance claim.'},
        'limits': ['Initial corrected model results were discarded because the adapter squeezed the native five-dimensional output. Fixed rank-contract tests and all 18 corrected model runs replace those results.',
            'The IEEE counterfactual changes only native triangle precision; the corrected counterfactual changes the complete triangle kernel, not just query rounding.',
            'Counterfactual H100 results are separately labeled and do not replace the frozen native acceptance reference.',
            'Cross-device RMSDs are rigorous intervals from frozen anchors, not newly measured exact cross-server coordinate distances.',
            'One cycle/step/sample, 1 warmup/3 repeats per mode. No full sampling, new training, MFU or 80% target acceptance claim.',
            'PPU FP32/TF32 kernels were not changed in this round; fresh timings do not establish a PPU optimization gain.']}
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# TF32 原生参考的模型级消融（2026-10-04）', '',
        f"本轮没有新增 PPU 提速。主要新证据是模型级精度定位：替换 H100 TriAttention 后，相对自身严格 FP32 的结构漂移减少比例中位数为 {summary['median_drift_reduction_pct']:.2f}%；RMSD≤0.1 Å 的组合从原生 TF32 的 4/18 增至自研诊断的 16/18，PPU 当前 TF32 也是 16/18。这些是各设备相对自身 FP32 的检查，不是跨设备验收通过率。", '',
        '置信度头补测消除了 base v0.5 两输入的 pLDDT 失败，六组合五项指标全部过 5% 差异门；结合 RMSD 区间，六组合为 3 PASS / 3 UNRESOLVED。正式冻结原生 TF32 参考仍为 3 PASS / 11 FAIL / 4 UNRESOLVED；项目整体精度和 80% 效率目标尚未通过。', '',
        '本轮完成九模型 × G01/G02 的四模式复测，以及 H100 的两条 TriAttention 诊断路径。所有四模式基线的预测哈希和五项置信度与上一轮完全相同；检查点、完整输入特征、原子顺序、外部资产和有序随机事件审计通过。原始张量和坐标保留在各自服务器。', '',
        '**原生参考保留不变。** H100_mixed_triangle_IEEE 仅在原生 TriAttention 调用内关闭 TF32，其余矩阵仍为 TF32；它是混合精度消融。H100_custom_triangle_RNE 使用已有、通过算子精度检查的 CUDA 一次舍入内核；它改变整个 TriAttention 实现，是自研诊断路径，不能称为 H100 原生 TF32，也不是只修 Q 舍入的实验。PPU 本轮没有修改算子。', '',
        '输入：G01=693 tokens、G02=245 tokens。cycle/step/sample=1，每模式 1 次预热、3 次测量。两个独立 worker 分别使用各设备的 GPU 0/7，同一组合的所有模式位于同一 GPU。本轮没有自己的训练任务并行；小百分比耗时差异不作为稳定收益。单位：秒。', '',
        table(timings), '',
        '下面是各设备 TF32 相对自身严格 FP32 的**直接实测** Kabsch RMSD，单位 Å。减少百分比描述替换 TriAttention 后的整体结构漂移变化，不能解释为单个舍入错误的独立贡献比例。', '',
        table(drifts), '',
        f"跨设备筛选：冻结原生参考 {summary['native_gate_counts']}；诊断参考 {summary['diagnostic_gate_counts']}。诊断门通过不能替代项目原生参考验收。", '',
        '跨设备 RMSD 仍使用已冻结的直接跨设备锚点及本轮各服务器直接实测的新旧预测距离，取三角不等式严格区间。门槛为 0.1 Å；五项置信度逐项相对差 ≤5%。区间跨越门槛时记 UNRESOLVED，不冒充精确距离。', '',
        table(structures), '',
        '全部五项置信度六列对照见 four_mode_and_diagnostic_confidence.csv；调用与精度入口记录见 ppu/derived.json、h100/derived.json。模型级测试没有重新测训练；前一轮通用反向缓存的训练结果应单独引用。', '',
        '置信度头补测覆盖 base v0.5、v2、mini_esm 各 G01/G02，共六组合。以 ConfidenceHead 类为通用边界，仅在其 forward 内关闭矩阵 TF32；自研 TriAttention 的显式 TF32 保持不变。六组合的预测坐标哈希均完全不变，检查点、完整输入和有序随机事件均通过。这里的 H100_custom_RNE_confidence_matmul_IEEE 是额外的混合精度诊断列；不能当作 H100 原生 TF32。', '',
        table(head_timings), '',
        table(head_confidences), '',
        f'六组合补测的诊断门计数：{dict(head_counts)}。两组 base v0.5 的 pLDDT 失败均被置信度头矩阵精度恢复消除；其他四项置信度也逐项检查。坐标不变，因此沿用同一 RMSD 严格区间；其余组合区间跨门槛仍记 UNRESOLVED。这是模块级消融，尚未隔离到头内单个 MM/BMM，也未推广为 PPU 正式策略或覆盖其余十二组合。', '',
        '全部十八组合的五项置信度对照：', '',
        table(confidences), '',
        '适配器首轮错误地去掉了 native low-level CUE 保留的 batch 轴，导致真实调用方 [0] 取走第一个序列组。虽然合成测试按 reshape 比较数值能通过，实际模型输出严重偏差。已修复输出 rank 契约，增加与原生返回 shape 的直接断言，并重跑全部 18 组自研诊断。首轮错误结果作废、仅留在服务器审计目录，不进入上述表格。', '',
        '此前原生算子位置控制已在冻结 CUE 0.8 和隔离 CUE 0.12 上复现：单个有效 key 从位置 0 移至 244，数学 score 应恒为 1，原生 TF32 却到 1.0078125；一次舍入的 PPU/CUDA 控制无该漂移。对应标量证据与来源哈希保存在 native_reference_prior_evidence.json。本轮用整个 TriAttention 的消融量化模型行为，不能把整内核差异全部归给 Q 漂移。', '',
        '本轮另提供不依赖模型或检查点的 reproduce_native_query_drift.py；H100 原生 CUE 0.8 实跑记录见 h100/native_query_reproducer.json。十六个 key 位置中，FP32 的 score/LSE 恒为 1，TF32 在末位置为 1.0078125。V 为常数时输出仍正确，说明只比较 attention 输出会漏检这一内部错误。', '',
        '原生 API 的 dtype、布局、mask、scale 与 TF32 全局开关契约参见 [NVIDIA Triangle Attention 文档](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_attention.html)。本轮默认软件包、项目正式入口和冻结基线均未修改。']
    (ROOT / 'report.md').write_text('\n'.join(lines)+'\n')
    print('NATIVE_REFERENCE_E2E_SUMMARY', summary['native_gate_counts'], summary['diagnostic_gate_counts'], summary['median_drift_reduction_pct'])


if __name__ == '__main__':
    main()
