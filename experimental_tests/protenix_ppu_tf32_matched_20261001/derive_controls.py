"""Summarize native H100 precision interventions without tensor export."""
import argparse
import hashlib
import json
from pathlib import Path
from derive_model import read, rmsd, coordinates, segments


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    args = p.parse_args()
    main = args.work/'model/G02'
    main_pair, main_bench = read(main/'paired.json'), read(main/'benchmark.json')
    base = {v: coordinates(next(r for r, m in zip(main_bench['records'], main_pair['order'])
                               if m == v and r['phase'] == 'repeat')) for v in ('fp32', 'tf32')}
    result = {'case': 'G02', 'scope': 'One-factor native H100 interventions; ordinary TF32 main baseline unchanged', 'experiments': {}}
    for group in ('init_control', 'cudnn_control', 'tf32_cudnn_control'):
        run = args.work/group/'G02'
        pair, bench = read(run/'paired.json'), read(run/'benchmark.json')
        for name in ('feature_summary.json', 'atom_identity.json', 'input_assets.json'):
            assert read(run/name) == read(main/name)
        a, b = segments(run/'rng_trace.jsonl'), segments(main/'rng_trace.jsonl')
        assert a[0] == b[0] and all(s == b[1] for s in a[1:])
        coords = {v: coordinates(next(r for r, m in zip(bench['records'], pair['order'])
                                    if m == v and r['phase'] == 'repeat')) for v in pair['summary']}
        modes = list(coords)
        result['experiments'][group] = {'summary': pair['summary'], 'input_rng_audit': 'PASS',
            'between_modes_rmsd_A': rmsd(coords[modes[0]], coords[modes[1]]),
            'vs_main_native_rmsd_A': {v: {m: rmsd(x, y) for m, y in base.items()} for v, x in coords.items()}}
    control = result['experiments']['cudnn_control']['summary']
    assert control['legacy_fp32_cudnn_tf32']['prediction_equal_reference']
    assert control['strict_fp32']['prediction_digest'] == main_pair['summary']['fp32']['prediction_digest']
    result['historical_fp32_cause_confirmed'] = True
    (args.work/'controls.json').write_text(json.dumps(result, indent=2)+'\n')
    print('NATIVE_CONTROLS_AUDIT_OK')


if __name__ == '__main__':
    main()
