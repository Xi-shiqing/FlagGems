"""Candidate public MM/mm_out, tail/stride/GEMV controls and real API times."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from statistics import median
import sys

HERE = Path(__file__).resolve().parent
sys.path.append('/mnt/cpfs/xishiqing/ppu224_container_mount/site-packages')
from runtime_guard_long_reduction import verify
imports = verify()
import torch
torch.set_num_threads(4)
parser = argparse.ArgumentParser()
parser.add_argument('--output-root', required=True, type=Path)
args = parser.parse_args()
matrix = importlib.import_module('flag_gems.ops.mm')
calls = []
original = matrix._long_reduction_splitk_mm


def observed(a, b, c, M, N, K):
    value = original(a, b, c, M, N, K)
    if value is not None:
        calls.append({'M_N_K': [M, N, K], 'parts': matrix._long_reduction_splitk_parts(a, b, c, M, N, K),
                      'TF32': bool(matrix._allow_tf32()), 'out': c is not None})
    return value


matrix._long_reduction_splitk_mm = observed


def same(a, b):
    assert a.shape==b.shape and a.dtype==b.dtype
    return a.detach().cpu().numpy().tobytes(order='C')==b.detach().cpu().numpy().tobytes(order='C')


def rne(x):
    bits = x.contiguous().view(torch.int32).to(torch.int64)
    return ((bits + 4095 + ((bits >> 13) & 1)) & 0xFFFFE000).to(torch.int32).view(torch.float32)


def metric(a, b):
    d, ref = a.double() - b.double(), b.double()
    return {'relative_L2': float(d.square().sum() / ref.square().sum().clamp_min(1e-300)) ** .5,
            'max_abs': float(d.abs().max()), 'elements': d.numel()}


def timed(function):
    value = function().cpu()
    assert same(value, function().cpu())
    observations = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); function(); b.record(); b.synchronize(); observations.append(a.elapsed_time(b))
    return value, {'median_device_ms': median(observations), 'all_five_device_ms': observations}


result = {'status': 'RUNNING_GENERIC_LONG_REDUCTION_MM_VALIDATION', 'generic_cases': [], 'real_cases': {},
          'actual_runtime_imports': imports, 'new_H100_execution': False, 'new_model_execution': False,
          'production_source_changed': False,
          'timing_scope': 'Both original and candidate public FlagGems MM API include their own allocations. Five device-event observations; no H100 timing, end-to-end or MFU claim.'}
for M, N, K, strided in [(1, 1, 8209, False), (17, 65, 8192, False), (17, 65, 8193, False),
                          (33, 17, 8209, True), (65, 33, 16387, True),
                          (64, 257, 10007, False), (512, 512, 8209, False)]:
    width = 2 if strided else 1
    ah = torch.sin(torch.arange(M * K * width, dtype=torch.float64).reshape(M, K * width) * .17).float()[:, ::width]
    bh = torch.cos(torch.arange(K * N * width, dtype=torch.float64).reshape(K, N * width) * .13).float()[:, ::width]
    a, b = ah.cuda(), bh.cuda()
    reference = ah.double() @ bh.double()
    quantized = rne(ah).double() @ rne(bh).double()
    for enabled in [False, True]:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        os.environ['FLAG_GEMS_PPU_LONG_REDUCTION_SPLITK'] = '0'
        baseline = matrix.mm(a, b).cpu()
        os.environ['FLAG_GEMS_PPU_LONG_REDUCTION_SPLITK'] = '1'
        before = len(calls)
        actual = matrix.mm(a, b).cpu()
        repeated = matrix.mm(a, b).cpu()
        assert same(actual, repeated) and torch.isfinite(actual).all()
        target = quantized if enabled and N != 1 else reference
        error = metric(actual, target)
        assert error['relative_L2'] <= (2e-4 if enabled else 2e-5), (M, N, K, enabled, error)
        if len(calls) == before:
            assert same(actual, baseline), 'Non-selected route changed'
        storage = torch.full((M, 2 * N), 123.5, device='cuda')
        out = storage[:, ::2]
        returned = matrix.mm_out(a, b, out=out)
        assert returned is out and same(out.cpu(), actual)
        assert bool((storage[:, 1::2] == 123.5).all())
        result['generic_cases'].append({'M_N_K': [M, N, K], 'strided_input': strided, 'TF32': enabled,
                                       'selected': len(calls) > before, 'error_vs_declared_CPU64_oracle': error,
                                       'mm_out_strided_and_guards': 'PASS', 'two_repeats_bitwise_equal': True})
    print('VALIDATED_GENERIC_MM', M, N, K, strided, flush=True)
operands_path = HERE / 'fixed_real_Triangle_Linear_weight_gradient_IEEE_precision/actual_Linear_weight_gradient_operands.pt'
package = torch.load(operands_path, map_location='cpu', mmap=True, weights_only=False)
result['actual_operand_file_sha256'] = hashlib.sha256(operands_path.read_bytes()).hexdigest()
for name, data in package.items():
    x, dy = data['x'], data['dy']
    a, b = dy.cuda().reshape(-1, dy.shape[-1]).t(), x.cuda().reshape(-1, x.shape[-1])
    ref = dy.reshape(-1, dy.shape[-1]).t().double() @ x.reshape(-1, x.shape[-1]).double()
    quantized = rne(dy).reshape(-1, dy.shape[-1]).t().double() @ rne(x).reshape(-1, x.shape[-1]).double()
    row = {}
    for enabled, precision in [(False, 'FP32'), (True, 'TF32')]:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        for switch, label in [('0', 'original'), ('1', 'candidate')]:
            os.environ['FLAG_GEMS_PPU_LONG_REDUCTION_SPLITK'] = switch
            value, observation = timed(lambda: matrix.mm(a, b))
            if switch == '0':
                saved = data['original_TF32_weight_gradient' if enabled else 'IEEE_weight_gradient']
                assert same(value, saved), 'Original real gradient reproduction failed'
            observation.update(vs_CPU64=metric(value, ref), vs_RNE_operand_CPU64=metric(value, quantized))
            row[precision + '/' + label] = observation
    result['real_cases'][name] = row
    (args.output_root / 'generic_long_reduction_MM_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('VALIDATED_REAL_MM_API', name, flush=True)
result.update(status='COMPLETE_GENERIC_LONG_REDUCTION_MM_VALIDATION', actual_selected_calls=calls)
(args.output_root / 'generic_long_reduction_MM_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
