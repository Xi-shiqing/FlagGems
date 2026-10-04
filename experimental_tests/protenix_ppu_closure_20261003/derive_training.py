"""Keep parameter tensors on their server; export complete scalar/hash audits."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text())


def signature(values):
    digest = hashlib.sha256()
    squared = 0.0
    maximum = 0.0
    for name, value in values.items():
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
        squared += float(value.double().square().sum())
        maximum = max(maximum, float(value.abs().max()))
        assert torch.isfinite(value).all(), name
    return {'sha256': digest.hexdigest(), 'l2_norm': squared**.5,
            'max_abs': maximum, 'finite': True, 'parameter_count': len(values)}


def compare(actual, reference):
    assert actual.keys() == reference.keys()
    difference, gold, maximum = 0.0, 0.0, 0.0
    for name, value in actual.items():
        delta = value.double() - reference[name].double()
        difference += float(delta.square().sum())
        gold += float(reference[name].double().square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {'global_relative_l2': (difference / max(gold, 1e-300))**.5,
            'global_max_abs': maximum, 'finite': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--rng-program', type=Path, required=True)
    parser.add_argument('--server', choices=('ppu', 'h100'), required=True)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--training-root', default='training')
    parser.add_argument('--output-name', default='derived_training.json')
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = {'status': 'PASS', 'server': args.server, 'cases': [],
              'rng_program_sha256': hashlib.sha256(args.rng_program.read_bytes()).hexdigest(),
              'steps': args.steps,
              'scope': 'Real AF3Trainer, all 48 Pairformer/4 MSA blocks. Parameter audit step and its overhead are explicitly recorded; only performance runs can be used for timing. All parameter tensors stay on their original server.'}
    modes = ('fp32', 'tf32', 'combined_tf32') if args.server == 'ppu' else ('fp32', 'tf32')
    for crop in ('131', '245'):
        baseline = None
        rows = {}
        for mode in modes:
            path = args.work / args.training_root / crop / mode
            assert (path / 'exit_code').read_text().strip() == '0', str(path)
            report = read(path / 'training_report.json')
            runtime = read(path / 'actual_training_entry.json')
            parameter_audit = read(path / 'actual_parameter_audit.json')
            assert report['status'] == 'pass', report.get('error')
            assert runtime['matmul_allow_tf32'] == (mode != 'fp32')
            assert len(report['losses']) == args.steps
            events = [json.loads(line) for line in (path / 'rng_trace.jsonl').read_text().splitlines()]
            for event in events:
                event.pop('index', None)
            row = {key: report[key] for key in ('protocol', 'environment', 'input_hashes', 'batch_sha256', 'batch_tensor_count',
                'batch_leaf_hashes', 'losses', 'gradient_norms', 'formal_event_median_ms', 'formal_wall_median_ms')}
            row.update({'runtime': runtime, 'parameter_audit': parameter_audit,
                        'rng_count': len(events), 'rng_semantic_sha256': hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()})
            if mode != 'combined_tf32':
                snapshot = torch.load(path / 'first_step_local_snapshot.pt', map_location='cpu', weights_only=True)
                row['initial_parameter_sha256'] = hashlib.sha256(json.dumps(snapshot['initial_sha256'], sort_keys=True).encode()).hexdigest()
                row['parameter_signatures'] = {kind: signature(snapshot[kind]) for kind in ('gradients', 'updates')}
                if mode == 'fp32':
                    baseline = snapshot
                else:
                    assert snapshot['initial_sha256'] == baseline['initial_sha256']
                    row['vs_local_fp32'] = {kind: compare(snapshot[kind], baseline[kind]) for kind in ('gradients', 'updates')}
                    del snapshot
            else:
                assert runtime['candidate_physical_backward_calls'] > 0
                assert parameter_audit['comparisons']
                row['vs_local_tf32'] = {kind: {k: v for k, v in metrics.items() if k != 'parameters'}
                    for kind, metrics in parameter_audit['comparisons'].items()}
                row['initial_parameter_sha256'] = rows['tf32']['initial_parameter_sha256']
            rows[mode] = row
            if mode != 'fp32':
                assert row['batch_leaf_hashes'] == rows['fp32']['batch_leaf_hashes']
                assert row['rng_semantic_sha256'] == rows['fp32']['rng_semantic_sha256']
            print('TRAIN_DERIVED', args.server, crop, mode, flush=True)
        del baseline
        result['cases'].append({'crop': int(crop), 'modes': rows})
    (args.work / args.output_name).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
