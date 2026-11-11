"""Complete registered boolean-index API, matched CPU bytes and every sample."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time

import flag_gems
import torch

p = argparse.ArgumentParser()
p.add_argument('--output-root', type=Path, required=True)
p.add_argument('--label', required=True)
args = p.parse_args()
assert flag_gems.vendor_name == 'thead'
args.output_root.mkdir(parents=True, exist_ok=True)
if (args.output_root / 'benchmark.json').exists():
    raise FileExistsError(args.output_root / 'benchmark.json')
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
bits = lambda x: x.contiguous().reshape(-1).view(torch.uint8)
torch.use_deterministic_algorithms(True)
records = []
with torch.no_grad(), flag_gems.use_gems():
    for tf32 in [False, True]:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = tf32
        shapes = [(n, width, axis) for n in [17, 245, 693, 2529]
                  for width in [3, 64, 257] for axis in [0, 1]] + [(4097, 3, 0), (4097, 3, 1)]
        for n, width, axis in shapes:
            shape = (n, width) if axis == 0 else (width, n)
            cpu = ((torch.arange(n * width, dtype=torch.int64) % 127) - 61).to(torch.float32).reshape(shape)
            gpu = cpu.to(flag_gems.device)
            for pattern in ['zero', 'one', 'mixed']:
                mask = torch.arange(n) % 3 != 0
                if pattern == 'zero':
                    mask.zero_()
                elif pattern == 'one':
                    mask.fill_(True)
                selected = mask.to(flag_gems.device)
                reference = torch.ops.aten.index.Tensor(cpu, [None] * axis + [mask])
                call = lambda: torch.ops.aten.index.Tensor(gpu, [None] * axis + [selected])
                assert torch.equal(bits(call().cpu()), bits(reference))
                for _ in range(10):
                    call()
                samples = {}
                for batch in [1, 16]:
                    values = []
                    for _ in range(7):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(batch):
                            call()
                        torch.cuda.synchronize()
                        values.append(1e6 * (time.perf_counter() - start) / batch)
                    samples[str(batch)] = values
                assert torch.equal(bits(call().cpu()), bits(reference))
                records.append({'matrix_precision': 'TF32' if tf32 else 'FP32',
                    'nmask': n, 'width': width, 'axis': axis, 'pattern': pattern,
                    'input_shape': list(gpu.shape), 'input_stride': list(gpu.stride()),
                    'input_sha256': hashlib.sha256(bits(cpu).numpy().tobytes()).hexdigest(),
                    'mask_sha256': hashlib.sha256(bits(mask).numpy().tobytes()).hexdigest(),
                    'CPU_bytes_equal': True, 'samples_mean_us_per_call': samples,
                    'median_us_per_call': {k: statistics.median(v) for k, v in samples.items()}})
result = {'status': 'PASS', 'label': args.label, 'rows': records, 'row_count': len(records),
          'source_sha256': sha(__file__), 'index_source': inspect.getsourcefile(flag_gems.index),
          'index_source_sha256': sha(inspect.getsourcefile(flag_gems.index)),
          'new_H100_execution': False}
assert len(records) == 156
with (args.output_root / 'benchmark.json').open('x') as handle:
    handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': 'PASS', 'rows': len(records)}))
