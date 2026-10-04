"""Real ConfidenceHead distance operands stay on server; export error/bin/timing scalars."""
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys

import torch
from runner import batch_inference
from runtime_precision import suspend_prediction_audit

ROOT = Path(__file__).resolve().parent
OUTPUT = Path(sys.argv[sys.argv.index('--paired-output')+1]).parent
initialize = batch_inference.get_default_runner
records = []


def probe(x, y, module, native):
    x, y = x.detach().clone(), y.detach().clone()
    assert x.shape[-1] == y.shape[-1] == 3
    gold = native(x.cpu().double(), y.cpu().double(), compute_mode='donot_use_mm_for_euclid_dist')
    lower, upper = module.lower_bins.cpu().double(), module.upper_bins.cpu().double()
    gold_bins = (gold[..., None] > lower) & (gold[..., None] < upper)
    row = {'shape': list(x.shape), 'dtype': str(x.dtype), 'torch': torch.__version__,
        'input_sha256': hashlib.sha256(bytes(x.cpu().contiguous().untyped_storage())).hexdigest(),
        'max_abs_coordinate': float(x.abs().max()), 'modes': {}}
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        for name, flag, method in [('mm_fp32', False, 'use_mm_for_euclid_dist'),
                ('mm_tf32', True, 'use_mm_for_euclid_dist'),
                ('direct_fp32', False, 'donot_use_mm_for_euclid_dist')]:
            torch.backends.cuda.matmul.allow_tf32 = flag
            value = native(x, y, compute_mode=method)
            for _ in range(3):
                native(x, y, compute_mode=method)
            samples = []
            for _ in range(20):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                native(x, y, compute_mode=method)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end)*1000)
            cpu = value.cpu().double()
            bins = (cpu[..., None] > lower) & (cpu[..., None] < upper)
            error = (cpu-gold).abs()
            row['modes'][name] = {'median_us': statistics.median(samples), 'samples_us': samples,
                'max_abs_error_A': float(error.max()), 'rms_error_A': float(error.square().mean().sqrt()),
                'changed_pair_bins_vs_fp64': int((bins != gold_bins).any(-1).sum()),
                'pair_count': cpu.numel(), 'max_self_distance_A': float(value.diagonal(dim1=-2, dim2=-1).abs().max())}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    records.append(row)


def get_runner(*a, **kw):
    runner = initialize(*a, **kw)
    for module in runner.model.modules():
        if type(module).__name__ != 'ConfidenceHead':
            continue
        original, code = module.forward, module.memory_efficient_forward.__func__.__code__
        def forward(*a, _original=original, _code=code, _module=module, **kw):
            if records:
                return _original(*a, **kw)
            native = torch.cdist
            def cdist(*values, **options):
                result = native(*values, **options)
                if not records and sys._getframe(1).f_code is _code:
                    with suspend_prediction_audit():
                        probe(values[0], values[1], _module, native)
                return result
            torch.cdist = cdist
            try:
                return _original(*a, **kw)
            finally:
                torch.cdist = native
        module.forward = forward
    return runner


batch_inference.get_default_runner = get_runner
try:
    sys.argv[0] = str(ROOT/'precision_entry.py')
    runpy.run_path(str(ROOT/'precision_entry.py'),run_name='__main__')
finally:
    (OUTPUT/'distance_probe.json').write_text(json.dumps({'records': records,
        'scope':'First warmup only; real representative-atom coordinates retained on server. Twenty CUDA event samples after three warmups per distance algorithm. No raw operands exported.'},indent=2)+'\n')
