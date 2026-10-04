"""Derive exact same-server structure metrics; export scalars and hashes only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', choices=('ppu', 'h100'), required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--prior-root', type=Path, required=True)
    parser.add_argument('--vendor', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.prior_root))
    from derive_matrix import coordinates, rmsd, segments
    import torch
    torch.set_num_threads(8)
    prior = read(args.prior_root / ('derived_matrix_serial.json' if args.server == 'ppu' else 'derived_matrix.json'))
    previous = {(r['case'], r['model']): r for r in prior['cases']}
    result = {'status': 'PARTIAL' if args.partial else 'PASS', 'server': args.server, 'cases': [],
        'scope': 'Exact same-server Kabsch RMSDs. Only scalar/hash evidence exported; raw predictions and annotations remain on original servers.'}
    for (case, model), old in previous.items():
        run = args.work / 'matrix' / case / model
        marker = run / 'exit_code'
        if args.partial and (not marker.exists() or marker.read_text().strip() != '0'):
            continue
        correction = args.work / 'corrected_matrix' / case / model
        if args.server == 'h100' and args.partial and not (correction / 'paired.json').exists():
            continue
        assert marker.read_text().strip() == '0', str(run)
        pair, bench = read(run / 'paired.json'), read(run / 'benchmark.json')
        assert pair['status'] == 'PASS'
        assert list(pair['checkpoint_sha256'].values()) == old['checkpoint_sha256']
        records = {mode: next(r for r, m in zip(bench['records'], pair['order']) if m == mode and r['phase'] == 'repeat') for mode in pair['summary']}
        if args.server == 'h100':
            assert (correction / 'exit_code').read_text().strip() == '0'
            fixed, fixed_bench = read(correction / 'paired.json'), read(correction / 'benchmark.json')
            assert fixed['status'] == 'PASS' and fixed['physical_gpu'] == pair['physical_gpu']
            assert fixed['checkpoint_sha256'] == pair['checkpoint_sha256']
            for name in ('feature_summary.json', 'atom_identity.json', 'input_assets.json'):
                assert read(correction / name) == read(run / name), name
            fixed_events = segments(correction / 'rng_trace.jsonl')
            fixed_hashes = [hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest() for e in fixed_events]
            assert fixed_hashes[0] == old['rng']['initialization_semantic_sha256']
            assert all(x == old['rng']['prediction_semantic_sha256'] for x in fixed_hashes[1:])
            records['tf32_triangle_corrected'] = next(r for r in fixed_bench['records'] if r['phase'] == 'repeat')
            pair['summary']['tf32_triangle_corrected'] = fixed['summary']['tf32_triangle_corrected']
            pair['triangle_calls_first_warmup']['tf32_triangle_corrected'] = fixed['triangle_calls_first_warmup']['tf32_triangle_corrected']
        current = {mode: coordinates(row) for mode, row in records.items()}
        for mode in ('fp32', 'tf32'):
            assert pair['summary'][mode]['prediction_digest'] == old['summary'][mode]['prediction_digest']
            assert pair['summary'][mode]['five_confidence_metrics'] == old['summary'][mode]['five_confidence_metrics']
        feature, atoms, assets = [read(run / name) for name in ('feature_summary.json', 'atom_identity.json', 'input_assets.json')]
        assert feature['tensors'] == old['features']
        assert atoms['sha256'] == old['atom_identity_sha256']
        assert assets['canonical_input_sha256'] == old['canonical_input_sha256']
        assert sorted(x['file_sha256'] for x in assets['assets'].values() if 'file_sha256' in x) == old['asset_hashes']
        events = segments(run / 'rng_trace.jsonl')
        hashes = [hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest() for e in events]
        assert hashes[0] == old['rng']['initialization_semantic_sha256']
        assert all(x == old['rng']['prediction_semantic_sha256'] for x in hashes[1:])
        if args.server == 'h100':
            refs = args.vendor / 'matrix' / case / model
            if case == 'G02' and model == 'protenix_base_default_v1.0.0':
                refs = args.vendor / 'e2e' / case
            anchors = {}
            for mode in ('fp32', 'tf32'):
                record = next(r for r in read(refs / mode / 'benchmark.json')['records'] if r['phase'] == 'repeat')
                assert record['prediction_digest']['sha256'] == old['anchor_prediction_sha256'][mode]
                anchors[mode] = coordinates(record, refs / mode / 'prediction_repeat_1.pt')
            drift = {mode: {anchor: rmsd(value, gold) for anchor, gold in anchors.items()} for mode, value in current.items()}
        else:
            drift = {mode: old['rmsd_to_own_frozen_anchors_A'][mode] for mode in ('fp32', 'tf32')}
        runtime = read(run / 'runtime_precision.json')
        if args.server == 'h100':
            runtime['tf32_triangle_corrected'] = read(correction / 'runtime_precision.json')['tf32_triangle_corrected']
        result['cases'].append({
            'case': case, 'model': model, 'summary': pair['summary'], 'N_token': old['N_token'],
            'checkpoint_sha256': old['checkpoint_sha256'], 'atom_count': atoms['atom_count'],
            'features': feature['tensors'], 'atom_identity_sha256': atoms['sha256'],
            'canonical_input_sha256': assets['canonical_input_sha256'], 'asset_hashes': old['asset_hashes'],
            'rng': old['rng'], 'input_audit': 'PASS', 'frozen_baseline_predictions_retained': True,
            'physical_gpu': pair['physical_gpu'], 'triangle_calls_first_warmup': pair['triangle_calls_first_warmup'],
            'rmsd_to_own_frozen_anchors_A': drift,
            'anchor_prediction_sha256': old['anchor_prediction_sha256'],
            'rmsd_to_current_local_fp32_A': {mode: rmsd(value, current['fp32']) for mode, value in current.items()},
            'rmsd_to_current_local_tf32_A': {mode: rmsd(value, current['tf32']) for mode, value in current.items()},
            'runtime_precision': runtime,
            'initial_incorrect_rank_adapter_result': 'DISCARDED; fixed adapter retains native five-dimensional output.' if args.server == 'h100' else None})
        print('REFERENCE_E2E_DERIVED', args.server, case, model, flush=True)
    if not args.partial:
        assert len(result['cases']) == 18
    (args.work / ('derived_partial.json' if args.partial else 'derived.json')).write_text(json.dumps(result, indent=2)+'\n')
    print('REFERENCE_E2E_DERIVATION_COMPLETE', args.server, len(result['cases']), flush=True)


if __name__ == '__main__':
    main()
