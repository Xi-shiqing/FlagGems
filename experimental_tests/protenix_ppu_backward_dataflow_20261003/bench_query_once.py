"""Actual native/FlagGems TriangleAttention with matched synthetic inputs and FP64 gold."""
import argparse
from collections import Counter
import hashlib
import importlib
from importlib import metadata
import json
import os
import statistics
import time
from pathlib import Path

os.environ.pop('PROTENIX_TF32_CAPTURE_DIR', None)
os.environ['PROTENIX_TF32_RESTORE_MODULES'] = ''

import numpy as np
import torch

KEYS = ('q', 'k', 'v', 'bias2')
LR = 1e-3


def sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def metrics(actual, reference):
    actual, reference = actual.detach().cpu().double(), reference.detach().cpu().double()
    diff = actual - reference
    return {'relative_l2': float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference).clamp_min(1e-300)),
            'max_abs': float(diff.abs().max()), 'finite': bool(torch.isfinite(actual).all())}


def inputs(s, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    arrays = {key: rng.integers(-32768, 32768, size=(1, 1, s, 4, 32), dtype=np.int32).astype(np.float32) * 2.0**-16
              for key in ('q', 'k', 'v')}
    arrays['bias2'] = rng.integers(-32768, 32768, size=(1, 1, 4, s, s), dtype=np.int32).astype(np.float32) * 2.0**-16
    arrays['upstream'] = rng.integers(-32768, 32768, size=(1, 1, s, 4, 32), dtype=np.int32).astype(np.float32) * 2.0**-12
    return arrays


def dense_gold(arrays):
    params = [torch.from_numpy(arrays[key]).double().requires_grad_() for key in KEYS]
    q, k, v, bias = params
    qh, kh, vh = [x.permute(0, 1, 3, 2, 4) for x in (q, k, v)]
    output = (qh @ kh.transpose(-1, -2) + bias).softmax(-1) @ vh
    output = output.permute(0, 1, 3, 2, 4)
    grads = torch.autograd.grad(output, params, torch.from_numpy(arrays['upstream']).double())
    return output.detach(), [grad.detach() for grad in grads]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', choices=('ppu', 'h100'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=(131, 245, 384))
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.server == 'h100':
        from cuequivariance_ops_torch import triangle_attention as native
        native_version = metadata.version('cuequivariance-ops-torch-cu12')
        original = None
    else:
        original = importlib.import_module('flag_gems.ops.triangle_attention')
        native_version = None
    candidates = {name: importlib.import_module(('cuda_' if args.server == 'h100' else '') + name)
                  for name in ('round_off_diagnostic', 'prequant_inputs', 'cached_ds')}
    modes = [('fp32', original, False), ('tf32', original, True)] + [(name, module, True) for name, module in candidates.items()]
    result = {'status': 'RUNNING', 'server': args.server, 'device': torch.cuda.get_device_name(), 'native_version': native_version,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'candidate_manifest': json.loads((Path(__file__).parent / 'candidate_manifest.json').read_text()),
        'scope': 'Operator-level inference and first-order Q/K/V/bias2 gradients plus one SGD update. Synthetic operands repeat one seeded group N times; dimensions match model shapes, activations are not model captures. Scale=1, all keys valid. H100 fixed_* are custom kernels, not native. No raw tensors persisted.',
        'training': {'learning_rate': LR, 'gold': 'CPU FP64 forward/backward; one-round FP32 updated parameters', 'reset_before_each_timing': True},
        'timing': {'warmup': 3, 'repeats': args.repeats, 'includes': 'Public entry, CUDA events and synchronized wall; full forward/backward/SGD for training',
                   'order': 'Alternating forward and reversed mode order across repetitions'}, 'cases': []}

    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    for s in args.sizes:
        arrays = inputs(s, 101 + s)
        gold_out, gold_grads = dense_gold(arrays)
        case = {'shape': [1, s, s, 4, 32], 'seed': 101 + s, 'rng': 'NumPy PCG64 integer generation on CPU; no GPU RNG',
                'group_input_sha256': {key: sha(value) for key, value in arrays.items()}, 'modes': {}}
        params = [torch.from_numpy(arrays[key]).cuda().expand(1, s, s, 4, 32).contiguous().requires_grad_() if key != 'bias2'
                  else torch.from_numpy(arrays[key]).cuda().requires_grad_() for key in KEYS]
        initial = [param.detach().clone() for param in params]
        upstream = torch.from_numpy(arrays['upstream']).cuda().expand_as(params[0]).contiguous()
        bias1 = torch.zeros((1, s, 1, 1, s), device='cuda')
        mask = torch.ones((s, 1, 1, s), dtype=torch.bool, device='cuda')
        # Views prepared outside timing. No packing/permute copy is hidden in the native entry.
        native_qkv = [param.squeeze(0).permute(0, 2, 1, 3) for param in params[:3]]

        def run(module, enabled):
            torch.backends.cuda.matmul.allow_tf32 = enabled
            torch.backends.cudnn.allow_tf32 = enabled
            if args.server == 'h100' and module is None:
                value = native(*native_qkv, params[3].squeeze(0), mask, 1., return_aux=True)[0]
                if value.ndim == 4:
                    value = value.unsqueeze(0)
                value = value.permute(0, 1, 3, 2, 4)
                assert value.shape == params[0].shape, (value.shape, params[0].shape)
                return value
            return module.triangle_attention(*params[:3], bias1, params[3], precision='tf32' if enabled else 'ieee', layout='BNSHD')

        def train(module, enabled):
            output = run(module, enabled)
            grads = torch.autograd.grad(output, params, upstream)
            with torch.no_grad():
                for param, grad in zip(params, grads):
                    param.add_(grad, alpha=-LR)

        def reset():
            with torch.no_grad():
                for param, value in zip(params, initial):
                    param.copy_(value)

        for name, module, enabled in modes:
            reset()
            output = run(module, enabled)
            row = {'label': ('H100 native ' if args.server == 'h100' else 'PPU FlagGems public ') + name if name in ('fp32', 'tf32')
                   else args.server + ' custom ' + name, 'output': metrics(output, gold_out.expand_as(output)),
                   'output_sha256': sha(output.detach().cpu().numpy()), 'parameters': {}, 'timing_samples': {'inference': [], 'training': []}}
            try:
                grads = torch.autograd.grad(output, params, upstream)
                row['training_status'] = 'SUPPORTED'
            except RuntimeError as error:
                if args.server != 'h100' or name != 'fp32' or 'Full precision FP32 backward pass for triangle attention is not implemented' not in str(error):
                    raise
                row['training_status'] = 'NATIVE_FP32_BACKWARD_UNSUPPORTED'
                row['training_error'] = str(error)
                grads = ()
            for key, param, grad, g in zip(KEYS, params, grads, gold_grads):
                reference = (g * s if key == 'bias2' else g.expand_as(param)).contiguous()
                updated = param.detach().clone()
                updated.add_(grad.detach(), alpha=-LR)
                gold_updated = (param.detach().cpu().double() - LR * reference).float()
                row['parameters'][key] = {'gradient': metrics(grad, reference),
                    'updated_parameter': metrics(updated, gold_updated),
                    'realized_fp32_update': metrics(updated - param.detach(), gold_updated - param.detach().cpu()),
                    'gradient_sha256': sha(grad.detach().cpu().numpy())}
            case['modes'][name] = row
            del output, grads
            print('ACCURACY_DONE', args.server, s, name, row['output']['relative_l2'], flush=True)
        result['cases'].append(case)
        save()
        for phase in ('inference', 'training'):
            for name, module, enabled in modes:
                if phase == 'training' and case['modes'][name]['training_status'] != 'SUPPORTED':
                    continue
                for _ in range(3):
                    reset()
                    if phase == 'inference':
                        with torch.no_grad():
                            run(module, enabled)
                    else:
                        train(module, enabled)
            torch.cuda.synchronize()
            for repeat in range(args.repeats):
                for name, module, enabled in modes if repeat % 2 == 0 else reversed(modes):
                    if phase == 'training' and case['modes'][name]['training_status'] != 'SUPPORTED':
                        continue
                    reset()
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin = time.perf_counter()
                    start.record()
                    if phase == 'inference':
                        with torch.no_grad():
                            run(module, enabled)
                    else:
                        train(module, enabled)
                    end.record()
                    end.synchronize()
                    case['modes'][name]['timing_samples'][phase].append({'wall_ms': 1000 * (time.perf_counter() - begin), 'event_ms': start.elapsed_time(end)})
            for row in case['modes'].values():
                row.setdefault('timing_median', {})[phase] = {key: statistics.median(sample[key] for sample in row['timing_samples'][phase]) for key in ('wall_ms', 'event_ms')} if row['timing_samples'][phase] else None
            save()
        if args.profile:
            for name, module, enabled in modes:
                if case['modes'][name]['training_status'] != 'SUPPORTED':
                    continue
                reset()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    train(module, enabled)
                    torch.cuda.synchronize()
                durations, counts = Counter(), Counter()
                for event in prof.events():
                    if 'CUDA' in str(event.device_type):
                        counts[event.name] += 1
                        durations[event.name] += event.time_range.elapsed_us()
                case['modes'][name]['training_gpu_profile'] = sorted(
                    [{'name': key, 'count': counts[key], 'total_us': value} for key, value in durations.items()],
                    key=lambda row: row['total_us'], reverse=True)
            save()
        print('TRIANGLE_CASE_DONE', args.server, s, flush=True)
    result['status'] = 'EXECUTION_PASS'
    save()
    print('BACKWARD_DATAFLOW_BENCH_DONE', args.server, flush=True)


if __name__ == '__main__':
    main()
