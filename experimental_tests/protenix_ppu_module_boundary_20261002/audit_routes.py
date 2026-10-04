"""Replay executed BMM/SDPA entries on their original operands; export scalars only."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import traceback


def tensor_meta(value):
    cpu = value.detach().cpu().contiguous()
    return {'shape': list(value.shape), 'stride': list(value.stride()),
            'dtype': str(value.dtype),
            'sha256': hashlib.sha256(cpu.numpy().tobytes()).hexdigest()}


def metrics(value, reference):
    import torch
    # Installed FlagGems vector_norm supports float32, but not CUDA float64.
    dtype = torch.float32 if value.is_cuda else torch.float64
    a, b = value.detach().to(dtype), reference.detach().to(dtype)
    d = a - b
    return {'relative_l2': float(d.norm() / b.norm().clamp_min(1e-30)),
            'max_abs': float(d.abs().max()), 'rms': float(d.square().mean().sqrt()),
            'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all())}


def rne(value):
    import numpy as np
    import torch
    a = value.detach().float().cpu().contiguous().numpy().copy()
    bits = a.view(np.uint32)
    finite = (bits & np.uint32(0x7f800000)) != np.uint32(0x7f800000)
    rounded = (bits + np.uint32(0xfff) + ((bits >> 13) & 1)) & np.uint32(0xffffe000)
    bits[finite] = rounded[finite]
    return torch.from_numpy(a)


def output(value):
    return value[0] if isinstance(value, (tuple, list)) else value


@contextmanager
def audit_routes(destination, policy):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    counts, seen, rows = Counter(), set(), []

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(func)
            relevant = func == torch.ops.aten.bmm.default or 'scaled_dot_product' in name
            if not relevant:
                return func(*args, **kwargs)
            tensors = [x for x in args if isinstance(x, torch.Tensor)]
            key = json.dumps([name, [list(x.shape) for x in tensors]])
            counts[key] += 1
            teacher = func(*args, **kwargs)
            # One replay per executed signature, capped to keep instrumentation finite.
            if key in seen or len(rows) >= 16 or tensors[0].dtype != torch.float32:
                return teacher
            seen.add(key)
            previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            replay = {}
            try:
                for mode in ('fp32', 'tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
                    torch.backends.cudnn.allow_tf32 = mode == 'tf32'
                    replay[mode] = output(func(*args, **kwargs))
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
                record = {'op': name, 'arguments_schema': str(func._schema),
                          'teacher_flags': list(previous),
                          'inputs': [tensor_meta(x) for x in tensors],
                          'stack': [f'{f.filename}:{f.lineno}:{f.name}' for f in traceback.extract_stack()
                                    if '/protenix/' in f.filename][-8:],
                          'tf32_vs_fp32': metrics(replay['tf32'], replay['fp32']),
                          'replay_vs_teacher': metrics(replay[policy], output(teacher)),
                          'output_hash': {mode: tensor_meta(y)['sha256'] for mode, y in replay.items()}}
                assert record['replay_vs_teacher']['max_abs'] == 0, record
                if func == torch.ops.aten.bmm.default:
                    a, b = [x.detach().cpu() for x in args[:2]]
                    # Keep the full reduction dimension, sample two batches and 8x8 outputs.
                    idx = [0] if a.shape[0] == 1 else [0, a.shape[0] - 1]
                    aa, bb = a[idx, :8, :], b[idx, :, :8]
                    gold = aa.double() @ bb.double()
                    rounded_gold = rne(aa).double() @ rne(bb).double()
                    record['sample_gold'] = {mode: {
                        'vs_ieee_fp64': metrics(y.detach().cpu()[idx, :8, :8], gold),
                        'vs_rne_tf32_fp64': metrics(y.detach().cpu()[idx, :8, :8], rounded_gold)}
                        for mode, y in replay.items()}
                record['kernel_trace'] = {}
                for mode in ('fp32', 'tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
                    torch.backends.cudnn.allow_tf32 = mode == 'tf32'
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        func(*args, **kwargs)
                        torch.cuda.synchronize()
                    record['kernel_trace'][mode] = [
                        {'name': event.name, 'device': str(event.device_type)}
                        for event in prof.events() if 'CUDA' in str(event.device_type)]
                rows.append(record)
                print('ROUTE_PROBE', policy, name, [list(x.shape) for x in tensors],
                      record['tf32_vs_fp32'], flush=True)
            finally:
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
            return teacher

    with Trace():
        yield
    destination[policy] = {'scope': 'first warmup only; original teacher outputs returned; no tensor export',
                           'replays': rows,
                           'counts': [json.loads(k) + [v] for k, v in counts.items()]}
