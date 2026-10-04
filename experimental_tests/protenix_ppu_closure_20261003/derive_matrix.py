"""Derive scalars on original servers; tensors and atom annotations stay there."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

MODELS = ('protenix-v2', 'protenix_base_20250630_v1.0.0', 'protenix_base_constraint_v0.5.0',
          'protenix_base_default_v0.5.0', 'protenix_base_default_v1.0.0', 'protenix_mini_default_v0.5.0',
          'protenix_mini_esm_v0.5.0', 'protenix_mini_ism_v0.5.0', 'protenix_tiny_default_v0.5.0')


def read(path):
    return json.loads(path.read_text())


def rmsd(a, b):
    if torch.equal(a, b):
        return 0.0
    p, q = [x.numpy().astype(np.float64).reshape(-1, 3) for x in (a, b)]
    assert p.shape == q.shape
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vh = np.linalg.svd(p.T @ q)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vh))
    return float(np.sqrt(np.mean(np.sum((p @ u @ correction @ vh - q)**2, axis=-1))))


def coordinates(record, path=None):
    value = torch.load(path or record['prediction_path'], map_location='cpu', weights_only=True).contiguous()
    assert hashlib.sha256(value.numpy().tobytes()).hexdigest() == record['prediction_digest']['sha256']
    assert torch.isfinite(value).all()
    return value


def segments(path):
    values = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        event.pop('index')
        if event['kind'] == 'rng.seed':
            values.append([])
        assert values
        values[-1].append(event)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', choices=('ppu', 'h100'), required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--vendor', type=Path, required=True)
    parser.add_argument('--matrix-root', default='matrix')
    parser.add_argument('--output-name', default='derived_matrix.json')
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = {'server': args.server, 'status': 'PASS', 'cases': [], 'scope': 'All 9 models and 2 inputs. Derived scalar/hash evidence only; original coordinates remain on their server.'}
    for case in ('G01', 'G02'):
        for model in MODELS:
            run = args.work / args.matrix_root / case / model
            assert (run / 'exit_code').read_text().strip() == '0'
            pair, bench = read(run / 'paired.json'), read(run / 'benchmark.json')
            assert pair['status'] == 'PASS'
            refs = args.vendor / ('full_gems' if args.server == 'ppu' else 'matrix') / case / model
            if args.server == 'h100' and case == 'G02' and model == 'protenix_base_default_v1.0.0':
                refs = args.vendor / 'e2e' / case
            current_records = {mode: next(r for r, m in zip(bench['records'], pair['order']) if m == mode and r['phase'] == 'repeat') for mode in pair['summary']}
            current = {mode: coordinates(r) for mode, r in current_records.items()}
            anchors, anchor_records = {}, {}
            for mode in ('fp32', 'tf32'):
                record = next(r for r in read(refs / mode / 'benchmark.json')['records'] if r['phase'] == 'repeat')
                anchors[mode] = coordinates(record, refs / mode / 'prediction_repeat_1.pt')
                anchor_records[mode] = record['prediction_digest']['sha256']
            feature, atoms, assets = [read(run / name) for name in ('feature_summary.json', 'atom_identity.json', 'input_assets.json')]
            for mode in ('fp32', 'tf32'):
                old_feature, old_atoms, old_assets = [read(refs / mode / name) for name in ('feature_summary.json', 'atom_identity.json', 'input_assets.json')]
                assert feature['tensors'] == old_feature['tensors']
                assert atoms['sha256'] == old_atoms['sha256']
                assert assets['canonical_input_sha256'] == old_assets['canonical_input_sha256']
                assert sorted(x.get('file_sha256') for x in assets['assets'].values() if 'file_sha256' in x) == sorted(x.get('file_sha256') for x in old_assets['assets'].values() if 'file_sha256' in x)
            events, old_events = segments(run / 'rng_trace.jsonl'), segments(refs / 'tf32' / 'rng_trace.jsonl')
            assert events[0] == old_events[0]
            assert all(s == old_events[1] for s in events[1:])
            drift = {mode: {anchor: rmsd(value, reference) for anchor, reference in anchors.items()} for mode, value in current.items()}
            candidate_same = None
            if args.server == 'ppu':
                candidate_same = rmsd(current['combined_tf32'], current['tf32'])
                assert candidate_same == 0
            result['cases'].append({'case': case, 'model': model, 'summary': pair['summary'], 'candidate_calls': pair['candidate_calls'],
                'checkpoint_sha256': list(pair['checkpoint_sha256'].values()), 'N_token': feature['N_token'], 'atom_count': atoms['atom_count'],
                'features': feature['tensors'], 'atom_identity_sha256': atoms['sha256'], 'canonical_input_sha256': assets['canonical_input_sha256'],
                'asset_hashes': sorted(x['file_sha256'] for x in assets['assets'].values() if 'file_sha256' in x),
                'rng': {'status': 'PASS', 'initialization_semantic_sha256': hashlib.sha256(json.dumps(events[0], sort_keys=True).encode()).hexdigest(),
                        'prediction_semantic_sha256': hashlib.sha256(json.dumps(old_events[1], sort_keys=True).encode()).hexdigest(),
                        'prediction_count': len(events)-1},
                'anchor_prediction_sha256': anchor_records, 'rmsd_to_own_frozen_anchors_A': drift,
                'candidate_rmsd_vs_current_tf32_A': candidate_same, 'input_audit_status': 'PASS',
                'local_tf32_vs_fp32_rmsd_A': rmsd(current['tf32'], current['fp32']),
                'runtime_precision': read(run / 'runtime_precision.json')})
            print('MATRIX_DERIVED', args.server, case, model, flush=True)
    (args.work / args.output_name).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
