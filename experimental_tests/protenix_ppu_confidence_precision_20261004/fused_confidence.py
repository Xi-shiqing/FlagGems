"""Use stable fused distance only at the semantic ConfidenceHead cdist call."""
from collections import Counter
import sys

import torch
from euclidean_distance import euclidean_distance

CALLS = Counter()


def install(model):
    matched = 0
    for name, module in model.named_modules():
        if type(module).__name__ != 'ConfidenceHead':
            continue
        matched += 1
        original, code = module.forward, module.memory_efficient_forward.__func__.__code__
        def forward(*a, _original=original, _code=code, _name=name, **kw):
            native = torch.cdist
            def distance(*values, **options):
                if sys._getframe(1).f_code is _code:
                    assert options.get('p', 2) == 2
                    assert len(values) == 2 and not (set(options)-{'p','compute_mode'})
                    CALLS[_name] += 1
                    return euclidean_distance(values[0], values[1])
                return native(*values, **options)
            torch.cdist = distance
            try:
                return _original(*a, **kw)
            finally:
                torch.cdist = native
        module.forward = forward
    assert matched
