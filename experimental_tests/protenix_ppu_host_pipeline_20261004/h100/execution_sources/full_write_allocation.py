"""Inference-only allocation for outputs proved fully written by their kernels."""
from collections import Counter
import torch

ENABLED = False
POISON = False
AUDIT = False
CALLS = Counter()


def empty(shape, *, dtype, device):
    if not ENABLED or torch.is_grad_enabled() or torch.device(device).type != 'cuda':
        return torch.empty(shape, dtype=dtype, device=device)
    # Reuse the allocator already verified for MM/BMM; keep global flags intact.
    from flag_gems.ops.tf32_precision import empty_full_write
    out = empty_full_write(shape, dtype=dtype, device=device)
    if POISON and out.numel():
        out.fill_(float('nan') if dtype.is_floating_point else 97 if dtype != torch.bool else True)
    if AUDIT:
        CALLS['allocations'] += 1
        CALLS['bytes'] += out.numel() * out.element_size()
    return out


def empty_like(template, *, dtype=None):
    dtype = template.dtype if dtype is None else dtype
    if (not ENABLED or torch.is_grad_enabled() or not template.is_cuda or
            template.layout != torch.strided or
            not torch.ops.aten.is_non_overlapping_and_dense(template)):
        return torch.empty_like(template, dtype=dtype)
    out = empty(template.shape, dtype=dtype, device=template.device)
    # Pointwise fast paths use physical order. Preserve dense transpose layouts.
    return out.as_strided(template.shape, template.stride())
