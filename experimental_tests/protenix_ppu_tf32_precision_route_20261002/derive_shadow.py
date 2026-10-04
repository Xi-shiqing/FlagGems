"""Confirm the early-Linear probes preserve the ordinary-TF32 model output."""
import argparse
import json
from pathlib import Path
from derive_model import read, segments


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', choices=('ppu', 'h100'), required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    args = p.parse_args()
    run, prior = args.work/'early_linear/G02', args.previous/'model/G02'
    bench, old_pair = read(run/'benchmark.json'), read(prior/'paired.json')
    mode = 'baseline_tf32' if args.server == 'ppu' else 'tf32'
    ref = old_pair['summary'][mode]
    assert len(bench['records']) == 2
    assert all(r['prediction_digest'] == ref['prediction_digest'] for r in bench['records'])
    assert all(r['five_confidence_metrics'] == ref['five_confidence_metrics'] for r in bench['records'])
    for name in ('feature_summary.json', 'atom_identity.json', 'input_assets.json'):
        assert read(run/name) == read(prior/name), name
    current, old = segments(run/'rng_trace.jsonl'), segments(prior/'rng_trace.jsonl')
    assert len(current) == 3 and current[0] == old[0]
    assert all(x == old[1] for x in current[1:])
    result = {'status': 'PASS', 'server': args.server, 'prediction_digest': ref['prediction_digest'],
        'predictions_bitwise_equal_ordinary_tf32': True, 'confidence_equal_ordinary_tf32': True,
        'input_features_assets_atom_order_equal': True, 'rng_ordered_events_equal': True,
        'initialization_events': len(current[0]), 'events_per_prediction': len(old[1]),
        'prediction_count': 2}
    (run/'integrity.json').write_text(json.dumps(result, indent=2)+'\n')
    print('SHADOW_INTEGRITY_PASS', args.server)


if __name__ == '__main__':
    main()
