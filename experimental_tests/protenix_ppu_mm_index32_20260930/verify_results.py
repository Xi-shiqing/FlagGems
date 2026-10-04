#!/usr/bin/env python3
"""Check the final MM candidate against archived operator/training controls."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    w = args.raw_root
    rows = []
    hashes = 0
    for path in sorted((w / 'live_v4').glob('*.candidate.json')):
        candidate = read(path)
        baseline = read(w / 'live' / path.name.replace('.candidate.', '.baseline.'))
        assert candidate['capture_input_sha256'] == baseline['capture_input_sha256']
        for section in ('operator_results', 'triangle_attention'):
            assert candidate[section].keys() == baseline[section].keys()
            for name, op in candidate[section].items():
                old = baseline[section][name]
                assert op['inputs'] == old['inputs'], (path.name, name, 'inputs')
                assert op['modes'].keys() == old['modes'].keys()
                for mode, result in op['modes'].items():
                    assert result['finite']
                    assert result['output_sha256'] == old['modes'][mode]['output_sha256'], (path.name, name, mode)
                    hashes += 1
                rows.append({'capture': path.name, 'op': name,
                             'precision_modes': list(op['modes']), 'bitwise_equal': True})
    assert len(rows) == 25 and hashes == 50, (len(rows), hashes)
    candidate = read(w / 'training_candidate_v4.json')
    controls = [read(w / name) for name in ('training_candidate.json', 'training_candidate_v3.json', 'training_current.json')]
    count = input_count = 0
    assert len(candidate['results']) == 4
    for i, result in enumerate(candidate['results']):
        for control in controls:
            old = control['results'][i]
            assert result['case'] == old['case']
            assert result['input_sha256'] == old['input_sha256']
            assert result['output_sha256'] == old['output_sha256'], result['case']
        input_count += len(result['input_sha256'])
        count += sum(len(v) for v in result['output_sha256'].values())
    assert count == 62 and input_count == 16, (count, input_count)
    pairs = read(w / 'index32_v4_paired.json')
    assert pairs['status'] == 'PASS' and len(pairs['rows']) == 9
    assert all(row['bitwise_equal'] for row in pairs['rows'])
    e2e = read(w / 'paired_full_gems/G02/paired.json')
    assert read(w / 'paired_full_gems/G02/benchmark.json')['environment']['use_flag_gems'] is True
    assert e2e['status'] == 'PASS' and e2e['bitwise_equal_predictions']
    result = {'status': 'PASS', 'candidate': 'candidate_v4',
              'mm_sha256': 'fe8686e66bab01824f5bfe28eba019ba9628a436cae5406ed6a65c3f987e7205',
              'operator_pairs': len(rows), 'operator_precision_output_hashes': hashes,
              'training_cases': 4, 'training_output_gradient_update_hashes': count,
              'training_input_hashes': input_count, 'same_process_model_predictions_equal': True,
              'paired_model_full_flag_gems': True,
              'paired_model_summary': e2e['summary'], 'rows': rows}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS', len(rows), 'operator pairs;', hashes, 'precision hashes;', count, 'training hashes')


if __name__ == '__main__':
    main()
