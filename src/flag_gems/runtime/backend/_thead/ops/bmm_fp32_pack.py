"""Copy large batch-interleaved FP32 operands before the unchanged BMM."""

import os
from functools import lru_cache

import torch

from flag_gems.ops.clone import clone

ENABLED = os.environ.get("FLAG_GEMS_PPU_BMM_FP32_PACK", "1") != "0"
MAX_WORKSPACE_BYTES = 1 << 30


def _dense(shape, strides):
    expected = 1
    for dim in sorted(range(3), key=lambda dim: strides[dim]):
        if shape[dim] == 1:
            continue
        if strides[dim] != expected:
            return False
        expected *= shape[dim]
    return True


@lru_cache(maxsize=512)
def eligible_layout(a_shape, a_stride, b_shape, b_stride):
    if len(a_shape) != 3 or len(b_shape) != 3:
        return False
    batch, m, k = a_shape
    if b_shape[:2] != (batch, k):
        return False
    n = b_shape[2]
    # Large interleaving wastes memory transactions across repeated dot tiles.
    # Bound copy cost and storage without identifying any model or exact shape.
    if batch < 32 or min(m, n) < 64 or k < 32:
        return False
    elements = batch * k * (m + n)
    if elements < 1 << 20 or 4 * elements > MAX_WORKSPACE_BYTES:
        return False
    if not _dense(a_shape, a_stride) or not _dense(b_shape, b_stride):
        return False
    return a_stride[0] == 1 or b_stride[0] == 1


def prepare(a, b, out=None):
    if (
        not ENABLED
        or not a.is_cuda
        or not b.is_cuda
        or a.device != b.device
        or a.dtype != torch.float32
        or b.dtype != torch.float32
        or a.is_neg()
        or b.is_neg()
    ):
        return None
    if not eligible_layout(tuple(a.shape), tuple(a.stride()), tuple(b.shape), tuple(b.stride())):
        return None
    if out is not None and (torch._C._overlaps(out, a) or torch._C._overlaps(out, b)):
        return None
    try:
        packed_a = a if a.is_contiguous() else clone(a, memory_format=torch.contiguous_format)
        packed_b = b if b.is_contiguous() else clone(b, memory_format=torch.contiguous_format)
    except torch.OutOfMemoryError:
        return None
    return packed_a, packed_b
