"""Whole allocating APIs: exact CPU reference, synchronized and batched latency."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time

import torch

import flag_gems


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(kind, n, dtype, layout):
    step = 2 if layout == 'stride2' else 1
    size = n * step
    if kind == 'nonzero':
        cpu = (torch.arange(size, dtype=torch.int64) % 3 != 0).to(dtype)
    else:
        cpu = ((torch.arange(size, dtype=torch.int64) * 17) % 11 - 5).to(dtype)
        if dtype == torch.int64 and size:
            cpu[0] = torch.iinfo(dtype).max
            if size > 1:
                cpu[1] = torch.iinfo(dtype).min
    gpu = cpu.to(flag_gems.device)
    return cpu[::step], gpu[::step]


def api(kind, tensor, option):
    if kind == 'sort':
        return torch.sort(tensor, stable=True, descending=option)
    if kind == 'unique':
        return torch.ops.aten._unique2(tensor, True, option, option)
    return (torch.nonzero(tensor),)


def equal(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a.cpu(), b.cpu())


def timed(call, batch):
    torch.cuda.synchronize()
    began = time.perf_counter()
    for _ in range(batch):
        call()
    torch.cuda.synchronize()
    return 1e6 * (time.perf_counter() - began) / batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--operators', nargs='+', choices=['sort', 'unique', 'nonzero'], default=['sort', 'unique'])
    args = p.parse_args()
    args.output_root.mkdir(exist_ok=True)
    assert flag_gems.vendor_name == 'thead'
    torch.use_deterministic_algorithms(True)
    cases = []
    for kind in args.operators:
        lengths = [17, 245, 693, 2529, 4096, 4097] if kind != 'nonzero' else [17, 245, 693, 2529, 8192, 8193]
        dtypes = [torch.int32, torch.int64] if kind == 'sort' else [torch.int64] if kind == 'unique' else [torch.bool, torch.int64]
        for n in lengths:
            for dtype in dtypes:
                for layout in ['dense', 'stride2'] if kind == 'sort' else ['dense']:
                    for option in [False, True] if kind != 'nonzero' else [False]:
                        cpu, gpu = inputs(kind, n, dtype, layout)
                        ref = api(kind, cpu, option)
                        cases.append((kind, n, dtype, layout, option, cpu, gpu, ref))
    records = []
    with torch.no_grad(), flag_gems.use_gems():
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = tf32
            for kind, n, dtype, layout, option, cpu, gpu, ref in cases:
                call = lambda: api(kind, gpu, option)
                equal(call(), ref)
                for _ in range(10):
                    call()
                samples = {str(batch): [timed(call, batch) for _ in range(7)] for batch in (1, 16)}
                equal(call(), ref)
                records.append({'kind': kind, 'n': n, 'dtype': str(dtype), 'layout': layout, 'option': option,
                                'input_shape': list(gpu.shape), 'input_stride': list(gpu.stride()),
                                'input_sha256': hashlib.sha256(cpu.contiguous().numpy().tobytes()).hexdigest(),
                                'matrix_precision': 'TF32' if tf32 else 'FP32', 'CPU_reference_exact': True,
                                'samples_mean_us_per_call': samples,
                                'median_us_per_call': {k: statistics.median(v) for k, v in samples.items()}})
    result = {'status': 'PASS', 'label': args.label, 'rows': records, 'row_count': len(records),
              'source_sha256': sha(__file__), 'gems': flag_gems.__file__,
              'sort_source': inspect.getsourcefile(flag_gems.sort),
              'sort_source_sha256': sha(inspect.getsourcefile(flag_gems.sort)),
              'nonzero_source_sha256': sha(inspect.getsourcefile(flag_gems.nonzero)),
              'scope': 'Complete allocating registered API; all seven samples retained. Batch 1 synchronizes each call; batch 16 amortizes outer synchronization only.',
              'new_H100_execution': False}
    with (args.output_root / 'benchmark.json').open('x') as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'rows': len(records), 'label': args.label}))


if __name__ == '__main__':
    main()
