"""Validate local inputs/RNG/predictions and compute scalar structural distances."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from derive_model import coordinates, read, rmsd, segments


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', choices=('ppu', 'h100'), required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--group', choices=('routes', 'interventions'), required=True)
    args = p.parse_args()
    run, prior = args.work/args.group/'G02', args.previous/'model'/'G02'
    pair = read(run/'routes.json')
    bench, old = read(run/'benchmark.json'), read(prior/'benchmark.json')
    frozen = read(prior/'paired.json')
    current_rng, old_rng = segments(run/'rng_trace.jsonl'), segments(prior/'rng_trace.jsonl')
    history = {'retained_failed_attempt_segments': 0, 'retained_failed_attempt_event_counts': []}
    initializations = [i for i, events in enumerate(current_rng) if events == old_rng[0]]
    if initializations == [0, 2]:
        # The first PPU instrumentation attempt failed on unsupported CUDA float64
        # vector_norm. Its appended trace remains intact; the retry is a new process.
        assert args.server == 'ppu' and args.group == 'routes'
        assert len(current_rng[1]) == 2 and current_rng[1] == old_rng[1][:2]
        history = {'retained_failed_attempt_segments': 2,
                   'retained_failed_attempt_event_counts': [488, 2],
                   'reason': 'Failed first diagnostic attempt: FlagGems CUDA float64 vector_norm unsupported. Original trace retained; successful retry starts at exact second initialization block.'}
        current_rng = current_rng[2:]
    else:
        assert initializations == [0]
    assert len(current_rng)-1 == len(bench['records'])
    assert current_rng[0] == old_rng[0]
    assert all(x == old_rng[1] for x in current_rng[1:])
    assert all(x == old_rng[1] for x in old_rng[1:])
    for file in ('feature_summary.json', 'atom_identity.json', 'input_assets.json'):
        assert read(run/file) == read(prior/file), file
    values, records = {}, {}
    for mode in pair['summary']:
        records[mode] = next(r for r, v in zip(bench['records'], pair['order'])
                             if v == mode and r['phase'] == 'repeat')
        values[mode] = coordinates(records[mode])
    previous_values = {}
    for mode in ('fp32', 'tf32'):
        old_mode = 'baseline_tf32' if args.server == 'ppu' and mode == 'tf32' else mode
        assert pair['summary'][mode]['prediction_digest'] == frozen['summary'][old_mode]['prediction_digest']
        assert pair['summary'][mode]['five_confidence_metrics'] == frozen['summary'][old_mode]['five_confidence_metrics']
        previous_values[mode] = coordinates(next(r for r, v in zip(old['records'], frozen['order'])
                                               if v == old_mode and r['phase'] == 'repeat'))
    checkpoint = read(args.previous/'derived.json')['checkpoint']
    digest = hashlib.sha256()
    with Path(checkpoint['path']).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(chunk)
    assert digest.hexdigest() == checkpoint['sha256']
    result = {'status': 'PASS', 'server': args.server, 'group': args.group,
        'checkpoint_sha256': digest.hexdigest(), 'input_audit': read(args.previous/'derived.json')['cases']['G02']['input_audit'],
        'rng_audit': {'status': 'PASS', 'initialization_events': len(current_rng[0]),
            'events_per_prediction': len(old_rng[1]), 'prediction_count': len(current_rng)-1,
            'initialization_semantic_sha256': hashlib.sha256(json.dumps(current_rng[0], sort_keys=True).encode()).hexdigest(),
            'prediction_semantic_sha256': hashlib.sha256(json.dumps(old_rng[1], sort_keys=True).encode()).hexdigest(),
            'trace_history': history},
        'baseline_predictions_equal_previous': True,
        'summary': pair['summary'],
        'rmsd_vs_local_baseline_A': {mode: {ref: rmsd(value, previous_values[ref])
                                          for ref in ('fp32', 'tf32')} for mode, value in values.items()},
        'all_local_pairwise_rmsd_A': {a+'_vs_'+b: rmsd(values[a], values[b])
            for i, a in enumerate(values) for b in list(values)[i+1:]},
        'scope': 'Scalar/hash JSON only; all raw activations and prediction tensors stay on their original server.'}
    (args.work/(args.group+'_derived.json')).write_text(json.dumps(result, indent=2)+'\n')
    print('DERIVED_ROUND_PASS', args.server, args.group, flush=True)


if __name__ == '__main__':
    main()
