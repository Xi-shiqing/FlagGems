"""Semantic confidence precision boundaries; no input/model/shape dispatch."""
from collections import Counter
from contextlib import contextmanager
import inspect
import sys

import torch

ACTIVE = [None]
CALLS = Counter()
SOURCES = {}


@contextmanager
def ieee():
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def install(model):
    matched = 0
    for name, module in model.named_modules():
        if type(module).__name__ != 'ConfidenceHead':
            continue
        matched += 1
        SOURCES[name] = inspect.getfile(module.forward)
        code = module.memory_efficient_forward.__func__.__code__
        for label in ('linear_no_bias_pae', 'linear_no_bias_pde'):
            child = getattr(module, label)
            original = child.forward
            def projection(*a, _original=original, _label=label, **kw):
                if ACTIVE[0] != 'tf32_confidence_outputs':
                    return _original(*a, **kw)
                CALLS[(ACTIVE[0], _label)] += 1
                with ieee():
                    return _original(*a, **kw)
            child.forward = projection
        original = module.forward
        def forward(*a, _original=original, _code=code, **kw):
            mode = ACTIVE[0]
            CALLS[(mode, 'head')] += 1
            if mode == 'tf32_confidence_all':
                with ieee():
                    return _original(*a, **kw)
            if mode not in ('tf32_confidence_outputs', 'tf32_confidence_distance'):
                return _original(*a, **kw)
            einsum, cdist = torch.einsum, torch.cdist
            def output(*values, **options):
                if values and values[0] == '...nc,ncb->...nb' and sys._getframe(1).f_code is _code:
                    CALLS[(mode, 'atom_output_einsum')] += 1
                    with ieee():
                        return einsum(*values, **options)
                return einsum(*values, **options)
            def distance(*values, **options):
                if sys._getframe(1).f_code is _code:
                    CALLS[(mode, 'cdist')] += 1
                    with ieee():
                        return cdist(*values, **options)
                return cdist(*values, **options)
            if mode == 'tf32_confidence_outputs':
                torch.einsum = output
            else:
                torch.cdist = distance
            try:
                return _original(*a, **kw)
            finally:
                torch.einsum, torch.cdist = einsum, cdist
        module.forward = forward
    assert matched, 'ConfidenceHead semantic boundary not found'


def audit():
    return {'matched_source_files': SOURCES,
        'actual_calls': [{'mode': mode, 'operation': op, 'count': count} for (mode, op), count in sorted(CALLS.items())],
        'scope': 'Inference-only synchronous experiment wrappers; global flags/functions always restored. Distance changes only ConfidenceHead cdist; outputs changes its two pair projections and two atom einsums. Triangle precision remains explicit TF32.'}
