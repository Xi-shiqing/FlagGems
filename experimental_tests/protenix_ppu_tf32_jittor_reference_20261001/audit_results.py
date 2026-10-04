#!/usr/bin/env python3
"""Calculate model quality on the original server; export derived metrics only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


MODEL = 'protenix_base_default_v1.0.0'


def read(path):
    return json.loads(path.read_text())


def coordinates(path, digest):
    value = torch.load(path, map_location='cpu', weights_only=True).numpy()
    assert hashlib.sha256(value.tobytes()).hexdigest() == digest['sha256']
    return value


def rmsd(a, b):
    p, q = [x.astype(np.float64).reshape(-1, 3) for x in (a, b)]
    assert p.shape == q.shape and np.isfinite(p).all() and np.isfinite(q).all()
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vh = np.linalg.svd(p.T @ q)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vh))
    return float(np.sqrt(np.mean(np.sum((p @ u @ correction @ vh - q)**2, axis=-1))))


def confidence(a, b):
    result = {}
    assert set(a) == set(b) == {'plddt', 'gpde', 'ptm', 'iptm', 'ranking_score'}
    for key in b:
        if a[key] is None or b[key] is None:
            assert a[key] is None and b[key] is None
            result[key] = {'defined': False, 'pass': True}
        else:
            delta = float(a[key] - b[key])
            result[key] = {'absolute': delta,
                           'relative_percent': abs(delta / b[key]) * 100 if b[key] else None,
                           'pass': abs(delta) <= .05 * abs(b[key]) if b[key] else delta == 0}
    return result


def rng_segments(path):
    segments = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        event.pop('index')  # Only the global counter changes with repeat count.
        if event['kind'] == 'rng.seed':
            segments.append([])
        assert segments
        segments[-1].append(event)
    return segments


def audit_inputs(run, frozen):
    for name in ('feature_summary.json', 'atom_identity.json', 'input_assets.json'):
        assert read(run / name) == read(frozen / name), (run, name)
    current, old = [rng_segments(p / 'rng_trace.jsonl') for p in (run, frozen)]
    assert current[0] == old[0]
    assert all(s == old[1] for s in current[1:])
    assert all(s == old[1] for s in old[1:])
    return {'status': 'PASS', 'features_atoms_assets_equal': True,
            'initialization_events': len(current[0]), 'prediction_count': len(current) - 1,
            'frozen_prediction_count': len(old) - 1, 'events_per_prediction': len(old[1]),
            'rng_rule': 'Every ordered event equal after removing only the index counter; raw trace lengths differ.'}


def comparison(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    idx = np.random.default_rng(101).integers(0, len(a), (10000, len(a)))
    draws = (1 - np.median(b[idx], axis=1) / np.median(a[idx], axis=1)) * 100
    return {'latency_reduction_percent': float((1 - np.median(b) / np.median(a)) * 100),
            'paired_bootstrap_95pct_ci': np.quantile(draws, [.025, .975]).tolist(),
            'bootstrap_seed': 101, 'bootstrap_draws': 10000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    w, v = args.work, args.work.parent / 'vendor_gemm_20260930'
    refs = read(args.references)
    result = {'status': 'PASS', 'source': 'original PPU server; predictions remain on server',
              'scope': 'InferenceRunner.predict; initialization/preprocessing excluded',
              'protocol': {'cycle': 1, 'step': 1, 'sample': 1, 'seed': 101,
                           'warmup': 3, 'total_repeats': 30, 'repeats_per_variant': 10},
              'limits': ['One checkpoint, two model inputs; operator captures include three inputs.',
                         'Not nine-model candidate acceptance, full default inference, full-model training or MFU.',
                         'Mixed precision retains strict diffusion and Triangle Attention; not all-TF32 inference.'],
              'model_pairs': [], 'operator_rows': []}
    for group in ('paired_models', 'paired_mixed', 'paired_unsplit'):
        for case in ('G01', 'G02'):
            p = w / group / case / MODEL
            pair, bench = read(p / 'paired.json'), read(p / 'benchmark.json')
            assert pair['full_flag_gems'] and bench['environment']['use_flag_gems']
            base = v / 'full_gems' / case / MODEL
            frozen = base / ('tf32' if group == 'paired_models' else 'diffusion_attention_ieee')
            audit = audit_inputs(p, frozen)
            mode_records = {name: [r for r, name2 in zip(bench['records'], pair['order'])
                                  if name2 == name and r['phase'] == 'repeat']
                            for name in pair['summary']}
            for name, records in mode_records.items():
                assert len(records) == 10
                assert len({r['prediction_digest']['sha256'] for r in records}) == 1
            strict = read(base / 'fp32/benchmark.json')
            strict_record = next(r for r in strict['records'] if r['phase'] == 'repeat')
            assert mode_records['fp32'][0]['prediction_digest'] == strict_record['prediction_digest']
            assert strict_record['prediction_digest']['sha256'] == refs[case]['ppu_fp32_prediction_sha256']
            candidate = mode_records['candidate_tf32'][0]
            if group == 'paired_unsplit':
                x = coordinates(Path(candidate['prediction_path']), candidate['prediction_digest'])
            else:
                rec = next(r for r in read(frozen / 'benchmark.json')['records'] if r['phase'] == 'repeat')
                assert candidate['prediction_digest'] == rec['prediction_digest']
                x = coordinates(frozen / 'prediction_repeat_1.pt', rec['prediction_digest'])
            y = coordinates(base / 'fp32/prediction_repeat_1.pt', strict_record['prediction_digest'])
            distance = rmsd(x, y)
            quality = {'rmsd_vs_ppu_fp32_A': distance,
                       'confidence_vs_ppu_fp32': confidence(candidate['five_confidence_metrics'], strict_record['five_confidence_metrics']),
                       'confidence_vs_h100_fp32': confidence(candidate['five_confidence_metrics'], refs[case]['confidence'])}
            old_h100 = w.parent / 'matched_h100_e2e_20260923/fp32' / case / MODEL
            old_rec = next(r for r in read(old_h100 / 'benchmark.json')['records'] if r['phase'] == 'repeat')
            if old_rec['prediction_digest']['sha256'] == refs[case]['prediction_sha256']:
                h = coordinates(old_h100 / 'prediction_repeat_1.pt', old_rec['prediction_digest'])
                quality['rmsd_vs_h100_fp32_A'] = rmsd(x, h)
                quality['h100_structure_method'] = 'Direct comparison; existing server reference hash matches frozen H100 strict output.'
                h100_distance = quality['rmsd_vs_h100_fp32_A']
            else:
                h100_distance = distance + refs[case]['ppu_fp32_vs_h100_fp32_rmsd_A']
                quality['rmsd_vs_h100_fp32_upper_bound_A'] = h100_distance
                quality['h100_structure_method'] = 'Triangle inequality for optimally aligned RMSD; bound, not directly measured distance.'
            quality['pass_rmsd_0_1A_confidence_5pct'] = distance <= .1 and h100_distance <= .1 and all(
                value['pass'] for key in ('confidence_vs_ppu_fp32', 'confidence_vs_h100_fp32') for value in quality[key].values())
            summary = pair['summary']
            row = {'case': case, 'experiment': group, 'summary': summary, 'input_rng_audit': audit,
                   'candidate_prediction_sha256': candidate['prediction_digest']['sha256'],
                   'strict_prediction_sha256': strict_record['prediction_digest']['sha256'],
                   'full_flag_gems': True, 'quality': quality,
                   'candidate_vs_previous': comparison(summary['baseline_tf32']['samples_seconds'], summary['candidate_tf32']['samples_seconds']),
                   'candidate_vs_strict_fp32': comparison(summary['fp32']['samples_seconds'], summary['candidate_tf32']['samples_seconds'])}
            result['model_pairs'].append(row)
            print('QUALITY', case, group, quality, flush=True)
    for filename, experiment in [('operator_paired.json', 'allocation'), ('operator_unsplit_paired.json', 'unsplit_stages3')]:
        report = read(w / filename)
        assert report['status'] == 'PASS' and len(report['rows']) == 32
        for row in report['rows']:
            if row['precision'] == 'fp32':
                assert row['bitwise_equal']
            samples = row['samples_wall_ms']
            result['operator_rows'].append({k: row[k] for k in ('capture', 'op', 'precision', 'bitwise_equal', 'medians_wall_ms')} |
                                          {'candidate_vs_baseline': row.get('candidate_vs_baseline'),
                                           'experiment': experiment, 'comparison': comparison(samples['baseline'], samples['candidate'])})
    result['pipeline_probe'] = read(w / 'unsplit_pipeline.json')
    result['rejected_stages_only_probe'] = read(w / 'pipeline.json')
    result['source_audit'] = read(w / 'source_audit.json')
    result['source_audit_unsplit'] = read(w / 'source_audit_unsplit.json')
    result['complete_write_coverage'] = {name: read(w / filename) for name, filename in
                                       [('allocation', 'full_write_coverage.json'), ('unsplit', 'full_write_unsplit_coverage.json')]}
    old, new = [read(w / name)['results'] for name in ('training.json', 'training_unsplit.json')]
    assert len(old) == len(new) == 4
    result['training'] = {'scope': 'Four standalone forward/backward/SGD probes; not full-model training',
                          'output_hash_count': 0, 'input_hash_count': 0, 'rows': []}
    for a, b in zip(old, new):
        assert a['case'] == b['case'] and a['input_sha256'] == b['input_sha256']
        assert a['output_sha256'] == b['output_sha256']
        assert all(error['finite'] for error in b['tf32_vs_fp32'].values())
        result['training']['input_hash_count'] += len(b['input_sha256'])
        result['training']['output_hash_count'] += sum(len(hashes) for hashes in b['output_sha256'].values())
        result['training']['rows'].append({k: b[k] for k in ('case', 'tf32_vs_fp32', 'timing')})
    result['training']['candidate_equals_previous_all_hashes'] = True
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for r in result['model_pairs']:
        print(r['case'], r['experiment'], {name: s['median_seconds'] for name, s in r['summary'].items()}, r['quality'])


if __name__ == '__main__':
    main()
