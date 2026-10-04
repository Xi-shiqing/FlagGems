"""Audit complete inputs/RNG on the original server; export only hashes/scalars."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', choices=('ppu', 'h100'), required=True)
    parser.add_argument('--stage', choices=('screen', 'screen_v2', 'matrix', 'recheck', 'steady', 'profile_candidate'), default='screen')
    parser.add_argument('--expected', type=int, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.parent/'closure_20261003'))
    from derive_matrix import segments
    previous = read(root.parent/'native_reference_e2e_20261004/derived.json')
    assert previous['status'] == 'PASS' and len(previous['cases']) == 18
    old = {(r['case'], r['model']): r for r in previous['cases']}
    result = {'server': args.server, 'status': 'PARTIAL' if args.partial else 'PASS', 'cases': [],
              'scope': 'All features, atom order, assets, checkpoint and ordered RNG audited on original server.'}
    for run in sorted((root/args.stage).glob('*/*')):
        if not (run/'paired.json').exists():
            continue
        key = (run.parent.name, run.name)
        reference = old[key]
        pair = read(run/'paired.json')
        assert pair['status'] == 'PASS'
        assert list(pair['checkpoint_sha256'].values()) == reference['checkpoint_sha256'], key
        feature, atoms, assets = [read(run/name) for name in
                                  ('feature_summary.json','atom_identity.json','input_assets.json')]
        assert feature['tensors'] == reference['features'], (key, 'features')
        assert atoms['sha256'] == reference['atom_identity_sha256'], (key, 'atoms')
        assert assets['canonical_input_sha256'] == reference['canonical_input_sha256'], (key, 'input')
        asset_hashes = sorted(x['file_sha256'] for x in assets['assets'].values() if 'file_sha256' in x)
        assert asset_hashes == reference['asset_hashes'], (key, 'assets')
        events = segments(run/'rng_trace.jsonl')
        assert digest(events[0]) == reference['rng']['initialization_semantic_sha256'], (key, 'initialization_rng')
        assert all(digest(e) == reference['rng']['prediction_semantic_sha256'] for e in events[1:]), (key, 'prediction_rng')
        assert len(events)-1 == len(pair['order'])
        for mode, row in pair['summary'].items():
            baseline = 'tf32' if mode in ('tf32_fullwrite', 'tf32_layout') else mode
            assert row['prediction_digest'] == reference['summary'][baseline]['prediction_digest'], (key, mode)
            assert row['five_confidence_metrics'] == reference['summary'][baseline]['five_confidence_metrics'], (key, mode)
        if args.server == 'ppu':
            gems = read(run/'gems_entry_audit.json')
            assert gems['status'] == 'PASS' and gems['full_flag_gems_requested'] and gems['calls']['direct_mm'] > 0
        result['cases'].append({'case': key[0], 'model': key[1], 'N_token': reference['N_token'],
            'summary': pair['summary'], 'physical_gpu': pair['physical_gpu'],
            'checkpoint_sha256': reference['checkpoint_sha256'],
            'feature_tensor_manifest_sha256': digest(feature['tensors']), 'atom_identity_sha256': atoms['sha256'],
            'canonical_input_sha256': assets['canonical_input_sha256'], 'asset_hashes': asset_hashes,
            'initialization_rng_sha256': digest(events[0]), 'prediction_rng_sha256': digest(events[1]),
            'prediction_count': len(events)-1, 'input_and_rng_audit': 'PASS',
            'original_predictions_and_all_five_confidence_retained': True,
            'allocation_counts_first_warmup': pair.get('allocation_counts_first_warmup', {}),
            'poison_first_warmup': pair.get('poison_first_warmup', False)})
    if not args.partial:
        assert len(result['cases']) == args.expected, (len(result['cases']), args.expected)
        assert (root/(args.stage+'.exit')).read_text().strip() == '0'
        if args.stage == 'recheck' and args.server == 'ppu':
            assert (root/'recheck_extra.exit').read_text().strip() == '0'
    name = 'derived_'+args.stage+('_partial' if args.partial else '')+'.json'
    (root/name).write_text(json.dumps(result, indent=2)+'\n')
    print('HOST_PIPELINE_AUDIT', args.server, args.stage, len(result['cases']), result['status'], flush=True)


if __name__ == '__main__':
    main()
