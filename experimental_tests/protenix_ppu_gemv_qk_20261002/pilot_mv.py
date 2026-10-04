"""Measure existing FlagGems MV before introducing another implementation."""
import json
import os
from pathlib import Path
import statistics
import sys
import time

if os.environ.get('PROTENIX_EXTRA_SITE_PACKAGES'):
    sys.path.append(os.environ['PROTENIX_EXTRA_SITE_PACKAGES'])
import torch
import flag_gems
from bench_tf32_training import _inputs, _sha

torch.set_num_threads(8)
torch.use_deterministic_algorithms(True)
root = Path(__file__).resolve().parent
arrays = _inputs('mm', (327680, 1, 16), 101)
rows = []
with flag_gems.use_gems(exclude=[]):
    for name, matrix, vector in (
        ('grad_activation', torch.from_numpy(arrays['upstream']), torch.from_numpy(arrays['weight'][:, 0])),
        ('grad_weight', torch.from_numpy(arrays['upstream']).T, torch.from_numpy(arrays['a'][:, 0])),
    ):
        gold = matrix.double() @ vector.double()
        a, v = matrix.cuda(), vector.cuda()
        row = {'name': name, 'matrix_shape': list(a.shape), 'matrix_stride': list(a.stride()),
               'operand_sha256': [_sha(matrix.numpy()), _sha(vector.numpy())], 'modes': {}}
        for mode in ('mm_fp32', 'mm_tf32', 'existing_mv'):
            torch.backends.cuda.matmul.allow_tf32 = mode == 'mm_tf32'
            fn = (lambda: torch.mv(a, v)) if mode == 'existing_mv' else (lambda: torch.mm(a, v[:, None])[:, 0])
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            samples = []
            for _ in range(10):
                start = time.perf_counter()
                value = fn()
                torch.cuda.synchronize()
                samples.append((time.perf_counter()-start)*1000)
            cpu = value.detach().cpu().double()
            row['modes'][mode] = {'median_ms': statistics.median(samples), 'samples_ms': samples,
                'relative_l2_vs_cpu_fp64': float((cpu-gold).norm()/gold.norm()),
                'max_abs_vs_cpu_fp64': float((cpu-gold).abs().max())}
        rows.append(row)
        print('MV_PILOT', row, flush=True)
(root/'pilot_mv.json').write_text(json.dumps({'status':'PASS', 'cases':rows}, indent=2)+'\n')
