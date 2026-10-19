"""Shape-guarded layout/batch compatibility probe; shared by both variants."""
import functools
import os
from collections import Counter
import torch

hits = Counter()
POLICY = os.environ.get('ATTENTION_COMPAT', 'none')
CAP = int(os.environ.get('ATTENTION_BATCH_CAP', '64'))


def wrap(function):
    if POLICY == 'none':
        return function

    @functools.wraps(function)
    def call(query, key, value, biases):
        if (torch.is_grad_enabled() or query.device.type != 'musa' or query.ndim != 4
                or query.shape[-2] <= 1024):
            return function(query, key, value, biases)
        if POLICY == 'contiguous':
            hits['contiguous_large'] += 1
            return function(query.contiguous(), key.contiguous(), value.contiguous(), biases)
        if POLICY != 'bounded':
            raise ValueError(POLICY)
        batch = query.shape[0]
        output = torch.empty_like(query, memory_format=torch.contiguous_format)
        for start in range(0, batch, CAP):
            end = min(start+CAP, batch)
            bs = [b[start:end] if b.ndim == 4 and b.shape[0] == batch else b for b in biases]
            output[start:end] = function(query[start:end].contiguous(), key[start:end].contiguous(),
                                        value[start:end].contiguous(), bs)
            hits['bounded_large_subcalls'] += 1
        return output
    return call


def install():
    from protenix.model.triangular import layers
    layers._attention = wrap(layers._attention)
