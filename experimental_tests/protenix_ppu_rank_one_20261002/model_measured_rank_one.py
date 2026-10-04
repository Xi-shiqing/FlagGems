"""IEEE outer products for TF32-enabled matrices with reduction length one."""
import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

try:
    from flag_gems.ops.tf32_precision import empty_full_write
except ImportError:
    empty_full_write = torch.empty


@libentry()
@triton.jit
def _outer_product(A, B, C, Bias, alpha, beta, COUNT: tl.constexpr,
                   M: tl.constexpr, N: tl.constexpr,
                   AB: tl.constexpr, AM: tl.constexpr,
                   BB: tl.constexpr, BN: tl.constexpr,
                   CB: tl.constexpr, CM: tl.constexpr, CN: tl.constexpr,
                   BM: tl.constexpr, BN_BIAS: tl.constexpr,
                   HAS_BIAS: tl.constexpr, WIDE: tl.constexpr,
                   BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if WIDE:
        index = index.to(tl.int64)
    mask = index < COUNT
    column = index % N
    row = (index // N) % M
    batch = index // (M * N)
    a = tl.load(A + batch * AB + row * AM, mask, other=0)
    b = tl.load(B + batch * BB + column * BN, mask, other=0)
    value = a * b
    if HAS_BIAS:
        value = value * alpha
        if beta != 0:
            bias = tl.load(Bias + row * BM + column * BN_BIAS, mask, other=0)
            value = value + bias * beta
    tl.store(C + batch * CB + row * CM + column * CN, value, mask)


def rank_one(a, b, out=None, *, bias=None, alpha=1, beta=1, enabled=False):
    """Return None for unsupported inputs; preserve the caller's FP32 path."""
    if not enabled or a.shape[-1] != 1:
        return None
    if (
        a.dtype != torch.float32 or b.dtype != torch.float32
        or a.device.type != "cuda" or b.device != a.device
        or a.ndim != b.ndim or a.ndim not in (2, 3)
        or b.shape[-2] != 1
        or (a.ndim == 3 and a.shape[0] != b.shape[0])
    ):
        return None
    if bias is not None and (bias.dtype != torch.float32 or bias.device != a.device):
        return None
    shape = (*a.shape[:-1], b.shape[-1])
    if out is None:
        out = empty_full_write(shape, dtype=a.dtype, device=a.device)
    else:
        if out.dtype != torch.float32 or out.device != a.device or out.shape != shape:
            return None
        extent = 1
        for stride, size in sorted((s, n) for s, n in zip(out.stride(), shape) if n > 1):
            if stride < extent:
                return None
            extent += (size - 1) * stride
        inputs = (a, b) if bias is None else (a, b, bias)
        if any(out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for x in inputs):
            return None
    if out.numel() == 0:
        return out
    bm = bn = 0
    if bias is not None:
        if a.ndim != 2 or bias.ndim > 2:
            return None
        sizes = (1,) * (2 - bias.ndim) + tuple(bias.shape)
        strides = (0,) * (2 - bias.ndim) + bias.stride()
        if any(size not in (1, target) for size, target in zip(sizes, shape)):
            return None
        bm, bn = (stride if size > 1 else 0 for size, stride in zip(sizes, strides))
    if min(*a.stride(), *b.stride()) < 0:
        return None
    ab, bb, cb = (a.stride(0), b.stride(0), out.stride(0)) if a.ndim == 3 else (0, 0, 0)
    wide = any(x.numel() > 2**31 - 1 or
               sum((n - 1) * s for n, s in zip(x.shape, x.stride())) > 2**31 - 1
               for x in (a, b, out))
    with torch_device_fn.device(a.device):
        _outer_product[(triton.cdiv(out.numel(), 512),)](
            a, b, out, bias, alpha, beta, out.numel(), a.shape[-2], b.shape[-1],
            ab, a.stride(-2), bb, b.stride(-1), cb, out.stride(-2), out.stride(-1),
            bm, bn, bias is not None, wide, 512, num_warps=4, enable_fp_fusion=False)
    return out
