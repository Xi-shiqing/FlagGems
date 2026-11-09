"""Keep the established reduction order while fusing temporary MM results."""
from functools import lru_cache

import torch
import triton

from flag_gems.ops.tf32_precision import empty_full_write
from flag_gems.runtime import torch_device_fn


@lru_cache(maxsize=256)
def _plan(m, n, k, am, ak, bk, bn):
    # The output already provides at least eight BM64/BN128 waves on PPU810.
    # Keep a bounded RHS and short reduction; no measured-shape table is used.
    if not (m >= 16384 and 64 <= k <= 1024 and 128 <= n <= 1024 and m * n >= 2**22):
        return False
    return (am, ak) == (k, 1) and (bk, bn) in ((n, 1), (1, k))


def _prepare_rhs(b, tf32):
    if not tf32 and b.is_contiguous():
        return b
    packed = empty_full_write(b.shape, dtype=b.dtype, device=b.device)
    if tf32:
        from .bmm_rne_layout import _round_flat, _round_transpose

        k, n = b.shape
        if b.is_contiguous():
            _round_flat[(triton.cdiv(k * n, 8192),)](
                b, packed, k * n, 8192, num_warps=8
            )
        else:
            _round_transpose[(triton.cdiv(k, 64), triton.cdiv(n, 64), 1)](
                b, packed, k, n, *b.stride(), 0, n, 1, 0, 64, num_warps=4
            )
    else:
        from flag_gems import copy_

        copy_(packed, b)
    return packed


def try_mm(a, b, out, split_k, tf32, direct_kernel, offsets_fit_i32):
    if (
        not a.is_cuda or not b.is_cuda or a.device != b.device
        or a.dtype != torch.float32 or b.dtype != torch.float32
        or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]
        or a.is_neg() or b.is_neg()
        or not _plan(a.shape[0], b.shape[1], a.shape[1], *a.stride(), *b.stride())
    ):
        return None
    m, k = a.shape
    n = b.shape[1]
    result = out if out is not None else empty_full_write(
        (m, n), dtype=a.dtype, device=a.device
    )
    # FP32 keeps BM/BN64; TF32 uses BN128 to amortize its input preparation.
    block_n = 128 if tf32 else 64
    with torch_device_fn.device(a.device):
        rhs = _prepare_rhs(b, tf32)
        use_i32 = tf32 and offsets_fit_i32(
            m, n, k, *a.stride(), *rhs.stride(), *result.stride(), split_k
        )
        direct_kernel[(triton.cdiv(m, 64) * triton.cdiv(n, block_n),)](
            a, rhs, result, m, n, k,
            *a.stride(), *rhs.stride(), *result.stride(),
            SPLIT_K=split_k, BLOCK_M=64, BLOCK_N=block_n, BLOCK_K=32,
            ALLOW_TF32=tf32, B_RNE_PREPACKED=tf32, INDEX_64=not use_i32,
            num_warps=4, num_stages=1,
        )
    return result
