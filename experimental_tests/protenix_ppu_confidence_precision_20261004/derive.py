"""Validate full audits on original servers; export small scalar/hash evidence."""
import argparse
from collections import Counter
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
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.parent / 'closure_20261003'))
    from derive_matrix import segments
    old = read(root.parent / 'native_reference_e2e_20261004/derived.json')
    assert old['status'] == 'PASS' and len(old['cases']) == 18
    result = {'status': 'PARTIAL' if args.partial else 'PASS', 'server': args.server, 'cases': [],
        'scope': 'All input features, assets, atom order, checkpoints and ordered RNG audited on original server. Confidence candidates retain exact coordinates; only scalar/hash evidence exported.'}
    for previous in old['cases']:
        case, model = previous['case'], previous['model']
        run = root / 'matrix' / case / model
        if args.partial and (not (run / 'exit_code').exists() or (run / 'exit_code').read_text().strip() != '0'):
            continue
        assert (run / 'exit_code').read_text().strip() == '0', str(run)
        pair = read(run / 'paired.json')
        assert pair['status'] == 'PASS'
        assert list(pair['checkpoint_sha256'].values()) == previous['checkpoint_sha256']
        feature, atoms, assets = [read(run / name) for name in ('feature_summary.json', 'atom_identity.json', 'input_assets.json')]
        assert feature['tensors'] == previous['features']
        assert atoms['sha256'] == previous['atom_identity_sha256']
        assert assets['canonical_input_sha256'] == previous['canonical_input_sha256']
        assert sorted(x['file_sha256'] for x in assets['assets'].values() if 'file_sha256' in x) == previous['asset_hashes']
        events = segments(run / 'rng_trace.jsonl')
        assert digest(events[0]) == previous['rng']['initialization_semantic_sha256']
        assert all(digest(e) == previous['rng']['prediction_semantic_sha256'] for e in events[1:])
        assert len(events)-1 == len(pair['order'])
        for mode in ('fp32', 'tf32'):
            assert pair['summary'][mode]['prediction_digest'] == previous['summary'][mode]['prediction_digest']
            assert pair['summary'][mode]['five_confidence_metrics'] == previous['summary'][mode]['five_confidence_metrics']
        baseline = 'tf32' if args.server == 'ppu' else 'tf32_triangle_corrected'
        assert pair['summary'][baseline]['prediction_digest'] == previous['summary'][baseline]['prediction_digest']
        assert pair['summary'][baseline]['five_confidence_metrics'] == previous['summary'][baseline]['five_confidence_metrics']
        audit = read(run / 'confidence_precision_audit.json')
        count = {(r['mode'], r['operation']): r['count'] for r in audit['actual_calls']}
        for mode in pair['summary']:
            assert count[mode, 'head'] == 4
            if mode.startswith('tf32_confidence_'):
                assert pair['summary'][mode]['prediction_digest'] == pair['summary'][baseline]['prediction_digest']
        assert count['tf32_confidence_outputs', 'atom_output_einsum'] == 8
        assert count['tf32_confidence_outputs', 'linear_no_bias_pae'] == 4
        assert count['tf32_confidence_outputs', 'linear_no_bias_pde'] == 4
        assert count['tf32_confidence_distance', 'cdist'] == 4
        runtime = read(run / 'runtime_precision.json')
        entries = {}
        for mode, trace in runtime.items():
            by_flag = Counter()
            python_flags = Counter()
            for row in trace['aten_calls']:
                by_flag[str(row['allow_tf32'])] += row['count']
            for row in trace['python_entries']:
                if 'flag_gems/ops/' in row['file']:
                    python_flags[str(row['matmul_allow_tf32'])] += row['count']
            entries[mode] = {'aten_matmul_calls_by_allow_tf32': dict(by_flag),
                'flag_gems_entries_by_allow_tf32': dict(python_flags), 'source_sha256': trace['source_sha256']}
        assert entries['tf32_confidence_outputs']['aten_matmul_calls_by_allow_tf32'].get('False', 0) > 0
        assert entries['tf32_confidence_all']['aten_matmul_calls_by_allow_tf32'].get('False', 0) > 0
        if args.server == 'ppu':
            gems = read(run / 'gems_entry_audit.json')
            assert gems['status'] == 'PASS' and gems['full_flag_gems_requested'] and gems['calls']['direct_mm'] > 0
        result['cases'].append({'case': case, 'model': model, 'N_token': previous['N_token'],
            'summary': pair['summary'], 'checkpoint_sha256': previous['checkpoint_sha256'],
            'feature_tensor_manifest_sha256': digest(feature['tensors']), 'atom_identity_sha256': atoms['sha256'],
            'canonical_input_sha256': assets['canonical_input_sha256'], 'asset_hashes': previous['asset_hashes'],
            'rng': {'initialization_semantic_sha256': digest(events[0]), 'prediction_semantic_sha256': digest(events[1]),
                'prediction_count': len(events)-1, 'status': 'PASS'},
            'input_and_rng_audit': 'PASS', 'native_predictions_and_metrics_retained': True,
            'candidate_coordinates_unchanged': True, 'physical_gpu': pair['physical_gpu'],
            'confidence_precision_audit': audit, 'runtime_precision_compact': entries,
            'triangle_calls_first_warmup': pair['triangle_calls_first_warmup']})
        print('CONFIDENCE_DERIVED', args.server, case, model, flush=True)
    if not args.partial:
        assert len(result['cases']) == 18
        assert (root / 'scope_control.exit').read_text().strip() == '0'
        assert set((root / 'matrix.exit').read_text().split()) == {'0'}
        result['worker_exit_markers'] = (root / 'matrix.exit').read_text().strip()
    (root / ('derived_partial.json' if args.partial else 'derived.json')).write_text(json.dumps(result, indent=2)+'\n')
    print('CONFIDENCE_AUDITS_COMPLETE', args.server, len(result['cases']))


if __name__ == '__main__':
    main()
