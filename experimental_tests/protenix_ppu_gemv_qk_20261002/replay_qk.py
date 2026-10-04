"""Replay live Pairformer operands in memory, persist scalars/hashes only."""
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import time
import traceback
from audit_routes import metrics, rne, tensor_meta


@contextmanager
def replay(destination, baseline=None, kind='pairformer'):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from controls import controls, switch
    entries = controls(baseline) if baseline is not None else []
    switch(entries, False)
    rows = []

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            teacher = func(*args, **kwargs)
            limit = 2 if kind == 'pairformer' else 1
            entry = torch.ops.aten.bmm.default if kind == 'pairformer' else torch.ops.aten.mm.default
            if func != entry or len(rows) == limit:
                return teacher
            frames = traceback.extract_stack()
            if kind == 'pairformer' and not (any(x.filename.endswith('/pairformer.py') for x in frames) and
                    any(x.filename.endswith('/primitives.py') and x.name == '_attention' for x in frames)):
                return teacher
            a, b = args[:2]
            if a.dtype != torch.float32:
                return teacher
            if kind == 'vector' and not (a.shape[1]>1 and (a.shape[0]==1 or b.shape[1]==1)):
                return teacher
            role = ('qk' if not rows else 'pv') if kind == 'pairformer' else 'vector_mm'
            previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            cpu_a, cpu_b = a.detach().cpu(), b.detach().cpu()
            gold = cpu_a.double() @ cpu_b.double()
            rounded_gold = rne(cpu_a).double() @ rne(cpu_b).double()
            row = {'role':role, 'inputs':[tensor_meta(x) for x in (a,b)],
                   'stack':[f'{x.filename}:{x.lineno}:{x.name}' for x in frames if '/protenix/' in x.filename],
                   'teacher_output':tensor_meta(teacher), 'teacher_flags':list(previous), 'modes':{},
                   'gold_ieee_fp64_sha256':tensor_meta(gold)['sha256'],
                   'gold_rne_fp64_sha256':tensor_meta(rounded_gold)['sha256']}
            try:
                modes = ('fp32','tf32','candidate_tf32') if entries else ('fp32','tf32')
                for mode in modes:
                    switch(entries, mode == 'candidate_tf32')
                    torch.backends.cuda.matmul.allow_tf32 = mode != 'fp32'
                    torch.backends.cudnn.allow_tf32 = mode != 'fp32'
                    for _ in range(3):
                        func(*args, **kwargs)
                    torch.cuda.synchronize()
                    samples = []
                    for _ in range(10):
                        start = time.perf_counter()
                        value = func(*args, **kwargs)
                        torch.cuda.synchronize()
                        samples.append((time.perf_counter()-start)*1000)
                    actual = value.detach().cpu()
                    row['modes'][mode] = {'median_ms':statistics.median(samples), 'samples_ms':samples,
                        'output_sha256':tensor_meta(actual)['sha256'],
                        'vs_ieee_fp64':metrics(actual,gold), 'vs_rne_fp64':metrics(actual,rounded_gold)}
                    if mode == 'tf32':
                        assert tensor_meta(value)['sha256'] == row['teacher_output']['sha256']
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                        func(*args, **kwargs)
                        torch.cuda.synchronize()
                    row['modes'][mode]['kernel_trace'] = [x.name for x in prof.events() if 'CUDA' in str(x.device_type)]
            finally:
                switch(entries, False)
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
            rows.append(row)
            print('PAIRFORMER_REPLAY',role,[list(x.shape) for x in (a,b)],flush=True)
            return teacher

    with Trace():
        yield
    switch(entries, False)
    assert len(rows) == (2 if kind == 'pairformer' else 1), rows
    if kind == 'pairformer':
        assert rows[0]['teacher_output']['shape'] == rows[1]['inputs'][0]['shape']
    Path(destination).write_text(json.dumps({'status':'PASS','records':rows,
        'scope':'Full actual QK/PV operands replayed only in original-server memory. Only error/timing/hash JSON persisted; original teacher outputs returned.'},indent=2)+'\n')
