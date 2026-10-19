"""Recompute headline results from archived scalar records; no GPU dependencies."""
import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def close(a, b):
    assert a is not None and b is not None
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def verify():
    groups = read('measurements.json')['groups']
    quality = read('quality.json')['records']
    summary = read('研究分析.json')
    count = sum(sum(r['phase'] == 'measured' for r in g['measurement']['rows'])
                for g in groups if g['counted_in_647'])
    assert count == 647
    assert len(quality) == 843 and all(x['status'] == 'PASS' for x in quality)
    primary = [g for g in groups if g['primary']]
    assert len(primary) == 42

    def select(mode, gpu, pdb, seed):
        values = [g for g in groups if (g['mode'], g['gpu'], g['pdb'], g['seed']) == (mode, gpu, pdb, seed)]
        assert len(values) == 1, (mode, gpu, pdb, seed)
        return values[0]

    def measured(group):
        return [r for r in group['measurement']['rows'] if r['phase'] == 'measured']

    def quality_values(group):
        return [q for q in quality if (q['mode'], q['gpu_visibility'], q['pdb'], q['seed'], q['kind']) ==
                (group['mode'], group['gpu'], group['pdb'], group['seed'], 'raw_generation0')]

    def verify_pair(row, native_mode, fused_mode):
        a = select(native_mode, row['native_gpu'], row['pdb'], row['seed'])
        b = select(fused_mode, row['fused_gpu'], row['pdb'], row['seed'])
        aa, bb = measured(a), measured(b)
        assert len(aa) == row['native_n'] and len(bb) == row['fused_n']
        assert a['gpu'] == b['gpu'] and a['features'] is not None and a['features'] == b['features']
        assert all(x['coordinates_finite'] and x['frozen_features'] for x in aa + bb)
        assert all(g['measurement']['amp_alignment'] and g['measurement']['mudnn_tf32'] for g in (a, b))
        am = st.median(x['predict_seconds'] for x in aa)
        bm = st.median(x['predict_seconds'] for x in bb)
        close(am, row['native_s']); close(bm, row['fused_s'])
        close(100 * (1 - bm / am), row['time_reduction_pct'])
        peak_a = max(x['peak_allocated_bytes'] for x in aa) / 1024 ** 3
        peak_b = max(x['peak_allocated_bytes'] for x in bb) / 1024 ** 3
        close(peak_a, row['native_gib']); close(peak_b, row['fused_gib'])
        close(100 * (1 - peak_b / peak_a), row['memory_reduction_pct'])
        qa, qb = quality_values(a), quality_values(b)
        assert len(qa) == len(aa) and len(qb) == len(bb)
        close(st.mean(q['complex_lddt'] for q in qa), row['native_lddt_mean'])
        close(st.mean(q['complex_lddt'] for q in qb), row['fused_lddt_mean'])
        close(row['fused_lddt_mean'] - row['native_lddt_mean'], row['lddt_delta'])

    for row in summary['main_pairs']:
        verify_pair(row, 'final_native', 'final_hybrid')
    assert sum(x['quality_screen'] == 'PASS' for x in summary['main_pairs']) == 20
    assert [x['pdb'] for x in summary['main_pairs'] if x['quality_screen'] == 'REVIEW'] == ['7bb8']
    for row in summary['multiseed']:
        suffix = '' if row['seed'] == 101 else f'_seed{row["seed"]}'
        verify_pair(row, 'aligned_control_native' + suffix, 'hybrid_control_fused' + suffix)
    for row in summary['quality_review']['pairs']:
        if row['seed'] == 101:
            verify_pair(row, 'final_native', 'final_hybrid')
        else:
            verify_pair(row, f'quality_review_native_seed{row["seed"]}', f'quality_review_hybrid_seed{row["seed"]}')

    for row in summary['crossover']:
        g = next(g for g in groups if g['mode'] == 'hybrid_crossover' and g['pdb'] == row['pdb'])
        calls = measured(g)
        aa = [x['predict_seconds'] for x in calls if x['variant'] == 'native']
        bb = [x['predict_seconds'] for x in calls if x['variant'] == 'fused']
        assert len(aa) == len(bb) == 8
        close(st.median(aa), row['native_s']); close(st.median(bb), row['fused_s'])

    g = next(g for g in groups if g['mode'] == 'storage_crossover')
    calls = measured(g)
    base = st.median(x['predict_seconds'] for x in calls if x['variant'] == 'native')
    for row in summary['storage']:
        values = [x for x in calls if x['variant'] == row['variant']]
        assert len(values) == 6
        close(st.median(x['predict_seconds'] for x in values), row['median_s'])
        close(100 * (1 - row['median_s'] / base), row['time_reduction_pct'])

    for name, key in [('paired_aggregate', None), ('qualified_aggregate', 'PASS'), ('deduplicated_aggregate', 'dedup')]:
        stored = summary[name]
        rows = [x for x in summary['main_pairs'] if x['pdb'] in stored['inputs']]
        if key == 'PASS':assert all(x['quality_screen'] == 'PASS' for x in rows)
        aa = sum(x['native_s'] for x in rows); bb = sum(x['fused_s'] for x in rows)
        close(aa, stored['native_sum_median_s']); close(bb, stored['fused_sum_median_s'])
        close(100 * (1 - bb / aa), stored['summed_latency_reduction_pct'])

    with (ROOT / 'h100_baseline.csv').open() as f:
        h100 = {x['pdb']: x for x in csv.DictReader(f)}
    assert len(h100) == 21 and all(int(x['formal_calls']) == 6 and int(x['structures_per_call']) == 5 for x in h100.values())
    for row in summary['h100']:close(float(h100[row['pdb']]['median_s']), row['h100_s'])
    ablation = read('ablation.json')['records']
    assert len(ablation) == 160 and all(x['numerical_pass'] for x in ablation)
    for row in read('source_provenance.json')['files']:
        p = ROOT / row['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['published_sha256'], p
    result = dict(status='PASS', formal_musa_calls=count, primary_calls=252,
                  quality_records=len(quality), quality_screen='20 PASS / 1 REVIEW',
                  headline_time_reduction_pct=summary['paired_aggregate']['summed_latency_reduction_pct'],
                  scope='Arithmetic and archive consistency only; no new GPU execution or independent quality recalculation.')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    verify()
