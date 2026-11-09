"""Optional allocation for inference kernels that write every output element.

The extension uses PyTorch's normal CUDA allocator, without changing global
deterministic settings. Build with tools/build_full_write_allocator.py, then set
FLAG_GEMS_NATIVE_FULL_WRITE=1 before importing FlagGems. Callers with autograd
enabled keep their existing allocation paths.
"""

import os

import torch

ENABLED = os.getenv("FLAG_GEMS_NATIVE_FULL_WRITE", "0") == "1"

if ENABLED:
    from ._full_write_allocator import empty_full_write as _empty


def empty(shape, *, dtype, device):
    return _empty(shape, dtype, torch.device(device))


def empty_for_overwrite(shape, *, dtype, device):
    """Allocate a buffer whose producer writes every element before any read.

    Callers must establish this for temporary buffers as well as final outputs.
    Autograd and non-CUDA allocations retain the ordinary allocation path.
    """
    if ENABLED and not torch.is_grad_enabled() and torch.device(device).type == "cuda":
        return _empty(shape, dtype, torch.device(device))
    return torch.empty(shape, dtype=dtype, device=device)


def empty_like(template, *, dtype=None):
    dtype = template.dtype if dtype is None else dtype
    if template.is_contiguous():
        return _empty(template.shape, dtype, template.device)
    if (
        template.layout == torch.strided
        and torch.ops.aten.is_non_overlapping_and_dense(template)
    ):
        result = _empty(template.shape, dtype, template.device)
        return result.as_strided(template.shape, template.stride())
    return torch.empty_like(template, dtype=dtype)
