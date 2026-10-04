"""Compute audit and structural scalars on the original server only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def read(path):
    return json.loads(path.read_text())


def rmsd(a, b):
    p, q = [x.detach().cpu().numpy().astype(np.float64).reshape(-1, 3) for x in (a, b)]
    assert p.shape == q.shape and np.isfinite(p).all() and np.isfinite(q).all()
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vh = np.linalg.svd(p.T @ q)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vh))
    return float(np.sqrt(np.mean(np.sum((p @ u @ correction @ vh - q)**2, axis=-1))))


def coordinates(record):
    value = torch.load(record['prediction_path'], map_location='cpu', weights_only=True)
    assert hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() == record['prediction_digest']['sha256']
    return value


def segments(path):
    result = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        event.pop('index')
        if event['kind'] == 'rng.seed':
            result.append([])
        assert result
        result[-1].append(event)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', choices=('ppu', 'h100'), required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--vendor', type=Path, required=True)
    args = p.parse_args()
    result = {'server': args.server, 'scope': 'derived scalars and hashes only; prediction tensors remain on original server', 'cases': {}}
    checkpoint = read(args.vendor/'checkpoint_audit_current.json')['models']['protenix_base_default_v1.0.0']
    digest = hashlib.sha256()
    with Path(checkpoint['path']).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(chunk)
    assert digest.hexdigest() == checkpoint['sha256']
    result['checkpoint'] = checkpoint | {'current_hash_rechecked': True}
    for case in ('G02',):
        run = args.work / 'forward_model' / case
        pair, bench = read(run/'paired.json'), read(run/'benchmark.json')
        refs = args.vendor / ('full_gems/'+case+'/protenix_base_default_v1.0.0' if args.server == 'ppu'
                             else ('matrix/G01/protenix_base_default_v1.0.0' if case == 'G01' else 'e2e/'+case))
        records = {mode: next(r for r, v in zip(bench['records'], pair['order'])
                              if v == mode and r['phase'] == 'repeat') for mode in pair['summary']}
        frozen = refs / 'tf32'
        for name in ('feature_summary.json', 'atom_identity.json', 'input_assets.json'):
            assert read(run/name) == read(frozen/name), (case, name)
        current, old = segments(run/'rng_trace.jsonl'), segments(frozen/'rng_trace.jsonl')
        assert current[0] == old[0]
        assert all(s == old[1] for s in current[1:]) and all(s == old[1] for s in old[1:])
        values = {mode: coordinates(rec) for mode, rec in records.items()}
        distances = {}
        modes = list(values)
        for i, a in enumerate(modes):
            for b in modes[i+1:]:
                distances[a+'_vs_'+b] = rmsd(values[a], values[b])
        drifts = {}
        for mode, value in values.items():
            precision = 'fp32' if mode == 'fp32' else 'tf32'
            ref = next(r for r in read(refs/precision/'benchmark.json')['records'] if r['phase'] == 'repeat')
            ref['prediction_path'] = str(refs/precision/'prediction_repeat_1.pt')
            drifts[mode] = {'rmsd_vs_frozen_same_precision_A': rmsd(value, coordinates(ref)),
                           'frozen_prediction_sha256': ref['prediction_digest']['sha256']}
        feature = read(run/'feature_summary.json')
        atoms = read(run/'atom_identity.json')
        assets = read(run/'input_assets.json')
        result['cases'][case] = {'summary': pair['summary'], 'same_device_rmsd_A': distances,
            'drift_vs_frozen': drifts, 'rng_audit': {'status': 'PASS',
                'initialization_events': len(current[0]), 'events_per_prediction': len(old[1]),
                'initialization_semantic_sha256': hashlib.sha256(json.dumps(current[0], sort_keys=True).encode()).hexdigest(),
                'prediction_semantic_sha256': hashlib.sha256(json.dumps(old[1], sort_keys=True).encode()).hexdigest(),
                'prediction_count': len(current)-1, 'rule': 'only index counters removed; ordered events identical'},
            'input_audit': {'status': 'PASS', 'features_atoms_assets_equal_frozen': True,
                'N_token': feature['N_token'], 'features': feature['tensors'],
                'atom_count': atoms['atom_count'], 'atom_identity_sha256': atoms['sha256'],
                'canonical_input_sha256': assets['canonical_input_sha256'],
                'asset_hashes': sorted(x['file_sha256'] for x in assets['assets'].values()
                                       if 'file_sha256' in x)},
            'environment': bench['environment'], 'runtime_precision': read(run/'runtime_precision.json')}
    (args.work/'derived_forward_model.json').write_text(json.dumps(result, indent=2)+'\n')
    print('DERIVED_METRICS_OK', args.server, list(result['cases']))


if __name__ == '__main__':
    main()
