"""Tighten cross-device RMSD bounds using existing frozen local anchors."""
import argparse
import json
from pathlib import Path

import torch

from derive_matrix import coordinates, read, rmsd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--vendor', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = {'status': 'PASS', 'scope': 'Additional frozen policy anchors, matched inputs. Scalars/hashes only.', 'cases': []}
    for anchor in read(args.work / 'extra_anchor_index.json')['rows']:
        case, model, mode = [anchor[k] for k in ('case', 'model', 'mode')]
        local = args.vendor / 'full_gems' / case / model / mode.removeprefix('ppu_')
        if not (local / 'benchmark.json').exists():
            matches = [p.parent for p in args.vendor.rglob('benchmark.json')
                       if any(r.get('prediction_digest', {}).get('sha256') == anchor['prediction_sha256']
                              for r in read(p).get('records', []))]
            assert matches, (case, model, mode)
            local = matches[0]
        run = args.work / 'matrix' / case / model
        benchmark, pair = read(run / 'benchmark.json'), read(run / 'paired.json')
        reference_record = next(r for r in read(local / 'benchmark.json')['records'] if r['phase'] == 'repeat')
        reference = coordinates(reference_record, local / 'prediction_repeat_1.pt')
        assert reference_record['prediction_digest']['sha256'] == anchor['prediction_sha256']
        assert read(local / 'feature_summary.json')['tensors'] == read(run / 'feature_summary.json')['tensors']
        assert read(local / 'atom_identity.json')['sha256'] == read(run / 'atom_identity.json')['sha256']
        assert read(local / 'input_assets.json')['canonical_input_sha256'] == read(run / 'input_assets.json')['canonical_input_sha256']
        distances = {}
        for current_mode in ('fp32', 'tf32'):
            record = next(r for r, policy in zip(benchmark['records'], pair['order']) if policy == current_mode and r['phase'] == 'repeat')
            distances[current_mode] = rmsd(coordinates(record), reference)
        result['cases'].append({**anchor, 'rmsd_to_anchor_A': distances, 'input_audit': 'PASS'})
    (args.work / 'extra_anchor_distances.json').write_text(json.dumps(result, indent=2) + '\n')
    print('EXTRA_ANCHOR_DERIVED', len(result['cases']))


if __name__ == '__main__':
    main()
