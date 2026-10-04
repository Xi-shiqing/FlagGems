"""Record actual Python entries and ATen calls during one warmup per policy."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


@contextmanager
def audit_prediction(records, variant):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    calls = Counter()
    entries = Counter()
    files = set()
    previous_profile = sys.getprofile()

    def profile(frame, event, arg):
        if event != 'call':
            return
        code = frame.f_code
        filename, name = code.co_filename, code.co_name
        relevant = ('flag_gems/ops/' in filename and name in
                    {'mm', 'bmm', 'addmm', '_direct_mm', '_bmm_rne_once'})
        relevant |= (('tri_attention' in filename or 'triangle_attention' in filename or
                      'cuequivariance_torch/primitives/triangle' in filename)
                     and name not in {'<module>', '<lambda>'})
        if not relevant:
            return
        values = {}
        for key, value in frame.f_locals.items():
            if any(s in key.lower() for s in ('precision', 'tf32', 'scale', 'causal')):
                if value is None or isinstance(value, (str, bool, int, float)):
                    values[key] = value
        entries[json.dumps({'file': filename, 'function': name,
                            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                            'arguments': values}, sort_keys=True)] += 1
        files.add(filename)

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default,
                        torch.ops.aten.bmm.default):
                calls[json.dumps({'op': str(func), 'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                    'tensors': [{'shape': list(x.shape), 'stride': list(x.stride()),
                                 'dtype': str(x.dtype)} for x in args if isinstance(x, torch.Tensor)]},
                    sort_keys=True)] += 1
            return func(*args, **kwargs)

    sys.setprofile(profile)
    try:
        with Trace():
            yield
    finally:
        sys.setprofile(previous_profile)
        packages = {}
        for package in ('torch', 'triton', 'cuequivariance-torch', 'cuequivariance-ops-torch-cu12'):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        records[variant] = {'scope': 'first warmup for this policy; measured repeats uninstrumented',
            'packages': packages, 'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'env': {k: v for k, v in os.environ.items() if k.startswith(('PROTENIX_TRIATTENTION',
                'PPU_FP32', 'FLAG_GEMS_PPU', 'CUBLAS_WORKSPACE'))},
            'aten_calls': [json.loads(k) | {'count': n} for k, n in calls.items()],
            'python_entries': [json.loads(k) | {'count': n} for k, n in entries.items()],
            'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              for p in sorted(files) if Path(p).is_file()}}
