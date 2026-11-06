"""Read-only native ATen observation and exact public-provider provenance."""
from collections import Counter
import hashlib
import inspect
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class MergedTrainingObserver(TorchDispatchMode):
    def __init__(self, kind):
        super().__init__()
        assert kind == 'accurate'
        self.calls = Counter()
        self.operations = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if any(isinstance(x, torch.Tensor) and x.is_cuda for x in args):
            self.operations[str(func)] += 1
        if func is torch.ops.aten._softmax.default and args[0].is_cuda:
            self.calls[(tuple(args[0].shape), int(args[1]) % args[0].ndim, (bool(args[2]) if len(args)>2 else bool(kwargs.get('half_to_float',False))))] += 1
        return func(*args, **kwargs)

    def evidence(self):
        import flag_gems
        root = Path(flag_gems.__file__).resolve().parent
        sources = {}
        for name in ['mm', 'addmm', 'bmm', 'linear', 'linear_backward', 'softmax', 'softmax_backward',
                     'mean', 'true_divide', 'pad', 'unfold_backward', 'index_select_backward']:
            fn = getattr(flag_gems, name)
            path = Path(inspect.getsourcefile(fn)).resolve()
            assert path.is_relative_to(root)
            sources[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        retained = {}
        for name in ['mm_fp32.py', 'bmm_fp32.py', 'addmm_fp32.py', 'softmax_fp32.py']:
            path = root/'runtime/backend/_thead/ops'/name
            retained[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        return {'public_sources': sources, 'retained_current_providers': retained,
                'actual_cuda_ATen_calls': dict(self.operations),
                'gradient_or_kernel_replacement_by_observer': False,
                'scope': 'Read-only actual ATen observation; merged package provides all kernels.'}
