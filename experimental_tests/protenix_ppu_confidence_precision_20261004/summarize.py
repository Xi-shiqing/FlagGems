"""Full four-mode comparisons with all experimental policies explicitly labeled."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
METRICS=('plddt','gpde','ptm','iptm','ranking_score')
POLICIES={'outputs':'tf32_confidence_outputs','distance_IEEE':'tf32_confidence_distance',
          'head_IEEE':'tf32_confidence_all','geometry_fast':'tf32_confidence_fused','geometry_RNE':'tf32_confidence_rne'}
BASE=('H100_native_FP32','H100_native_TF32','PPU_FlagGems_FP32','PPU_FlagGems_TF32')


def read(path):return json.loads(path.read_text())


def save_csv(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def table(rows):
    fields=list(rows[0])
    def cell(x):
        return f'{x:.6g}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join('---' for _ in fields)+' |']+
        ['| '+' | '.join(cell(row[k]) for k in fields)+' |' for row in rows])


def gate(p,h,interval):
    errors={m:100*abs(p[m]-h[m])/abs(h[m]) if h[m] else (0 if p[m]==0 else None) for m in METRICS}
    passing=all(x is not None and x<=5 for x in errors.values())
    status='FAIL' if not passing or interval['status']=='FAIL' else interval['status']
    return {'status':status,'confidence_pass':passing,'relative_errors_pct':errors,'rmsd_interval':interval}


def main():
    data={s:read(ROOT/s/'derived.json') for s in ('ppu','h100')}
    fused={s:read(ROOT/s/'derived_fused.json') for s in data}
    rne={s:read(ROOT/s/'derived_rne.json') for s in data}
    for collection in (data,fused,rne):
        assert all(x['status']=='PASS' and len(x['cases'])==18 for x in collection.values())
    cases={s:{(r['case'],r['model']):r for r in data[s]['cases']} for s in data}
    for side in data:
        for mode,collection in [('tf32_confidence_fused',fused),('tf32_confidence_rne',rne)]:
            for row in collection[side]['cases']:
                assert row['coordinate_hash_unchanged'] and row['input_and_rng_audit']=='PASS'
                cases[side][row['case'],row['model']]['summary'][mode]=row['summary']
    prior=read(ROOT/'prior_summary.json')
    anchors={(r['input'],r['model']):r for r in prior['cases']}
    timings,metrics,reports=[],[],[]
    counts={label:Counter() for label in POLICIES}
    confidence_counts=Counter()
    ppu_exact=0
    for key,p in cases['ppu'].items():
        h=cases['h100'][key]
        for field in ('checkpoint_sha256','feature_tensor_manifest_sha256','atom_identity_sha256','canonical_input_sha256','asset_hashes'):
            assert p[field]==h[field],(key,field)
        for field in ('initialization_semantic_sha256','prediction_semantic_sha256'):
            assert p['rng'][field]==h['rng'][field]
        ps,hs=p['summary'],h['summary']
        base={'input':key[0],'tokens':p['N_token'],'model':key[1]}
        baseline=[hs['fp32'],hs['tf32'],ps['fp32'],ps['tf32']]
        timing=base|{label+'_s':s['median_seconds'] for label,s in zip(BASE,baseline)}
        timing['H100_custom_triangle_RNE_s']=hs['tf32_triangle_corrected']['median_seconds']
        for label,mode in POLICIES.items():
            timing['PPU_FlagGems_TF32_'+label+'_s']=ps[mode]['median_seconds']
            timing['H100_custom_triangle_TF32_'+label+'_s']=hs[mode]['median_seconds']
        timing['PPU_TF32_vs_H100_native_TF32_time_ratio']=ps['tf32']['median_seconds']/hs['tf32']['median_seconds']
        timing['PPU_gpu'],timing['H100_gpu']=p['physical_gpu'],h['physical_gpu']
        timings.append(timing)
        gates={}
        interval=anchors[key]['cross_device_gates']['custom_triangle_rne']['rmsd']
        for label,mode in POLICIES.items():
            gates[label]=gate(ps[mode]['five_confidence_metrics'],hs[mode]['five_confidence_metrics'],interval)
            counts[label][gates[label]['status']]+=1
            confidence_counts[label]+=gates[label]['confidence_pass']
        exact=ps['tf32_confidence_rne']['five_confidence_metrics']==ps['tf32']['five_confidence_metrics']
        ppu_exact+=exact
        for metric in METRICS:
            row=base|{'metric':metric}|{label:s['five_confidence_metrics'][metric] for label,s in zip(BASE,baseline)}
            row['H100_custom_triangle_RNE']=hs['tf32_triangle_corrected']['five_confidence_metrics'][metric]
            for label,mode in POLICIES.items():
                row['PPU_TF32_'+label]=ps[mode]['five_confidence_metrics'][metric]
                row['H100_custom_triangle_TF32_'+label]=hs[mode]['five_confidence_metrics'][metric]
                row[label+'_relative_error_pct']=gates[label]['relative_errors_pct'][metric]
            metrics.append(row)
        reports.append(base|{'input_and_rng_audit':'PASS','all_candidate_coordinate_hashes_unchanged':True,
            'PPU_RNE_five_confidence_exactly_retained':exact,'diagnostic_gates':gates,
            'formal_native_gates_unchanged':anchors[key]['cross_device_gates']})
    save_csv('all_policies_timings.csv',timings)
    save_csv('all_policies_confidence.csv',metrics)
    operator=[]
    for case in ('G01','G02'):
        sides={s:read(ROOT/s/'rne_probe'/case/'protenix_base_default_v0.5.0/rne_probe.json')['records'][0] for s in data}
        row={'input':case,'tokens':int(case=='G01')*448+245}
        for side,prefix in [('h100','H100_native'),('ppu','PPU_FlagGems')]:
            record=sides[side]
            for mode,label in [('mm_fp32','FP32'),('mm_tf32','TF32'),('direct_fp32','direct_FP32'),('fused_fp32','fused_RNE')]:
                v=record['modes'][mode]
                column_prefix=('H100_experimental' if side=='h100' else 'PPU_FlagGems_experimental') if mode=='fused_fp32' else prefix
                for metric in ('median_us','max_abs_error_A','rms_error_A','changed_pair_bins_vs_fp64','max_self_distance_A'):
                    row[column_prefix+'_'+label+'_'+metric]=v[metric]
            row[prefix+'_fused_speedup_vs_TF32']=record['modes']['mm_tf32']['median_us']/record['modes']['fused_fp32']['median_us']
            row[prefix+'_actual_operand_sha256']=record['input_sha256']
        operator.append(row)
    save_csv('four_mode_distance_operator.csv',operator)
    selected_timing=[{k:r[k] for k in ('input','tokens','model',*[b+'_s' for b in BASE],
        'H100_custom_triangle_RNE_s','PPU_FlagGems_TF32_geometry_RNE_s','H100_custom_triangle_TF32_geometry_RNE_s')} for r in timings]
    selected_confidence=[{k:r[k] for k in ('input','model','metric',*BASE,'H100_custom_triangle_RNE',
        'PPU_TF32_geometry_RNE','H100_custom_triangle_TF32_geometry_RNE','geometry_RNE_relative_error_pct')} for r in metrics]
    scope=read(ROOT/'scope.json')
    summary={'status':'ALL_18_CASE_CONFIDENCE_PRECISION_AND_GENERIC_GEOMETRY_COMPLETE','cases':reports,
        'formal_native_gate_counts_unchanged':prior['native_gate_counts'],
        'diagnostic_gate_counts':{k:dict(v) for k,v in counts.items()},'five_confidence_pass_counts':dict(confidence_counts),
        'PPU_RNE_five_confidence_exactly_retained_count':ppu_exact,'operator_rows':operator,'scope':scope,
        'limits':['Formal frozen native reference is unchanged. Corrected H100 diagnostics are not native TF32.',
            f"Cross-device RMSD intervals are reused only after exact coordinate-hash checks; {counts['geometry_RNE']['UNRESOLVED']} unresolved intervals cannot be counted as passes.",
            'Original PPU distance already uses a direct FlagGems kernel and ignores compute_mode. API-request labels do not imply a TF32 GEMM on PPU.',
            'Operator probe uses each device own real activation coordinates and own CPU FP64 gold; operand hashes differ across devices.',
            'No new full-model training test or 80%/MFU acceptance. Native gradient fallback verified on small controls only.',
            'RNE fused phase runs separately from base sweep on same per-case physical GPU. Three wall-time repeats cannot certify small whole-model speed changes.']}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 通用置信度精度边界与欧氏距离优化（2026-10-04）','',
        '本轮完成九模型 × G01/G02 的新四模式基线、三条置信度精度消融，以及快速平方根和 RNE 平方根两版融合几何距离。最终候选使用 RNE。所有候选的坐标哈希与对应 TF32 基线完全一致，完整输入、外部资产、原子顺序、检查点和有序随机事件均通过。正式算子包与冻结原生参考没有修改。','',
        '关键定位：恢复最终评分投影的精度没有解决 base v0.5 的 pLDDT 失败；只恢复 ConfidenceHead 的 cdist 精度就解决了该问题。PPU 已有 FlagGems cdist 直接计算坐标差，不使用矩阵展开，也忽略 compute_mode，因此本轮在 PPU 上主要改进距离算子的执行效率。实际入口及源码哈希见 ppu/cdist_route_source.json。','',
        'H100 的矩阵展开使用 d²=||x||²+||y||²−2x·y；大坐标使消减误差被放大。真实 G01 坐标的自身距离在原生 TF32 下最大达到 172 Å，距离分箱有 332 个原子对改变。RNE 融合距离先减坐标，再平方求和，保留现有 PPU 的 round-to-nearest 平方根语义；一个执行块处理多个原子对，代替每个原子对独占一个执行块。参数选择基于特征维度及块内工作量，没有模型、输入或特殊 shape 白名单。','',
        '矩阵展开与距离计算模式的实现依据：[PyTorch 2.7.1 Distance.cpp](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/Distance.cpp)。源码支持上述机制判断；分箱变化及模型影响来自本轮实测。','',
        f"五项置信度全部过 5% 差异门的组合数：{dict(confidence_counts)}。各诊断路径的综合门计数：{summary['diagnostic_gate_counts']}。正式原生参考仍为 {prior['native_gate_counts']}。",'',
        f'最终 RNE 路径在 PPU 上五项置信度与原有 TF32 完全相同的组合数：{ppu_exact}/18。全部候选坐标不变，因此 RMSD 严格区间沿用上一轮，仍需精确跨设备距离才能处理待判定项。','',
        '输入 G01=693 tokens、G02=245 tokens，cycle/step/sample=1，各模式预热一次、重复三次。单位秒。H100_custom_triangle 列采用自研 RNE TriAttention，是额外诊断列；PPU 最终候选是完整 FlagGems 环境加实验融合 cdist，其余 MM/BMM/TriAttention 路径保持原有 TF32。','',
        table(selected_timing),'',
        '单算子探针：各设备自身真实代表原子坐标，对原机 CPU FP64 直接距离计算作参考，三次预热后二十次 CUDA event 样本。两端激活哈希不同，不能称为相同坐标张量的直接跨设备误差比较。原始坐标留在服务器。PPU 的 FP32/TF32 列表示外部开关，实际都进入同一已有 FlagGems 直接距离实现。完整误差、分箱和自身距离见 four_mode_distance_operator.csv。','',
        table([{k:r[k] for k in ('input','tokens','H100_native_FP32_median_us','H100_native_TF32_median_us',
            'PPU_FlagGems_FP32_median_us','PPU_FlagGems_TF32_median_us','H100_experimental_fused_RNE_median_us','PPU_FlagGems_experimental_fused_RNE_median_us')} for r in operator]),'',
        '全部五项置信度对照：','',table(selected_confidence),'',
        '其余投影 IEEE、距离 IEEE、整个置信度头 IEEE、快速平方根版的完整时间与置信度列见 all_policies_timings.csv、all_policies_confidence.csv。相对误差逐项以对应 H100 诊断路径为分母。','',
        '通用内核控制覆盖 1/3/5/32/129 维、广播、非连续输入、空集合和零维特征；梯度启用的输入回退原生直接距离，值和梯度在小规模控制中完全一致。没有新 Triton 反向实现，也没有重新验证完整模型训练。','',
        'H100 首轮一组 mini_ism G01 在 ESM 预处理阶段遇到共享文件竞争。失败原始目录保留在服务器 control_failures；加入首批 CPU 特征生成期间的文件锁后，该组重跑且完整输入审计通过。后续 H100 模型阶段采用该锁；GPU 推理不持锁。','',
        '距离算子是单独的局部收益，一次置信度头调用占整网时间很少。不能将其倍数解释为模型同等加速；本轮三次墙钟样本也不足以确认小百分比的整网收益。例如 v2 G02 的 PPU 原 TF32 为 2.320 秒，RNE 候选为 2.387 秒，本次反而慢约 2.9%；v2 G01 为 17.212→17.262 秒。当前不能排除运行波动或候选整网开销，也没有证据承诺整网免费收益。候选保留在实验目录，FP32 成果及当前 TF32 正式默认保持不变。']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print('ALL_CONFIDENCE_AND_GEOMETRY_SUMMARY_COMPLETE',summary['diagnostic_gate_counts'], 'PPU_EXACT',ppu_exact)


if __name__=='__main__':main()
