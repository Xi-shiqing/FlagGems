"""Bound long-K serial work and overlap loads inside parallel partitions."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.ops.tf32_precision import empty_full_write, round_tf32_rne
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.device_info import get_sm_count


@triton.jit
def partials_kernel(
    A, B, P, M, N, K, AM, AK, BK, BN,
    SPLITS: tl.constexpr,
    GROUPS: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
    TF32: tl.constexpr,
):
    pid = tle.program_id(0)
    group = pid % GROUPS
    tile = pid // GROUPS
    cols = tl.cdiv(N, TN)
    rm = ((tile // cols) * TM + tl.arange(0, TM)).to(tl.int64)
    rn = ((tile % cols) * TN + tl.arange(0, TN)).to(tl.int64)
    chunk = tl.cdiv(K, SPLITS)
    for split in range(group, SPLITS, GROUPS):
        begin = split * chunk
        end = tl.minimum(begin + chunk, K)
        acc = tl.zeros((TM, TN), tl.float32)
        for offset in range(0, chunk, TK):
            rk = (begin + offset + tl.arange(0, TK)).to(tl.int64)
            a = tl.load(
                A + rm[:, None] * AM + rk[None, :] * AK,
                (rm < M)[:, None] & (rk < end)[None, :],
                0.0,
            )
            b = tl.load(
                B + rk[:, None] * BK + rn[None, :] * BN,
                (rk < end)[:, None] & (rn < N)[None, :],
                0.0,
            )
            acc += tl.dot(
                round_tf32_rne(a, TF32),
                round_tf32_rne(b, TF32),
                out_dtype=tl.float32,
                allow_tf32=TF32,
            )
        tl.store(
            P + (split * M + rm[:, None]) * N + rn[None, :],
            acc,
            (rm < M)[:, None] & (rn < N)[None, :],
        )


@lru_cache(maxsize=256)
def _plan(m, n, k, sms):
    if k <= 8192 or not (32 <= m <= 1024 and 32 <= n <= 1024):
        return None
    # Preserve the existing long-MM eligibility. Use larger output tiles for
    # reuse, and partition K enough to bound serial work and fill the device.
    if triton.cdiv(m, 32) * triton.cdiv(n, 32) > 4 * sms:
        return None
    tiles = triton.cdiv(m, 64) * triton.cdiv(n, 64)
    needed = max(triton.cdiv(4 * sms, tiles), triton.cdiv(k, 8192))
    limit = min(64, k // 128, (64 * 1024**2) // (4 * m * n))
    if limit < 2:
        return None
    cap = 1 << (limit.bit_length() - 1)
    splits = min(cap, triton.next_power_of_2(needed))
    return splits, splits, 3, 64


def _execute(a, b, out, tf32, splits, groups, stages, tile=64):
    from .mm_fp32 import mm_direct_fused, mm_kernel_h100_fp32_splitk_reduce

    m, k = a.shape
    n = b.shape[1]
    result = (
        out
        if out is not None
        else empty_full_write((m, n), dtype=a.dtype, device=a.device)
    )
    tiles = triton.cdiv(m, tile) * triton.cdiv(n, tile)
    if groups == 1:
        mm_direct_fused[(tiles,)](
            a, b, result, m, n, k, *a.stride(), *b.stride(), *result.stride(),
            SPLIT_K=splits,
            BLOCK_M=tile,
            BLOCK_N=tile,
            BLOCK_K=32,
            ALLOW_TF32=tf32,
            B_RNE_PREPACKED=False,
            INDEX_64=True,
            num_warps=4,
            num_stages=stages,
        )
    else:
        partials = empty_full_write((splits, m, n), dtype=a.dtype, device=a.device)
        partials_kernel[(tiles * groups,)](
            a, b, partials, m, n, k, *a.stride(), *b.stride(),
            SPLITS=splits,
            GROUPS=groups,
            TM=tile,
            TN=tile,
            TK=32,
            TF32=tf32,
            num_warps=4,
            num_stages=stages,
        )
        mm_kernel_h100_fp32_splitk_reduce[(triton.cdiv(m * n, 256),)](
            partials, result, m * n, *result.stride(), n, splits,
            BLOCK=256,
            num_warps=4,
        )
    return result


def try_mm(a, b, out, tf32):
    if (
        not a.is_cuda
        or not b.is_cuda
        or a.device != b.device
        or a.dtype != torch.float32
        or b.dtype != torch.float32
        or a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[0]
        or a.is_neg()
        or b.is_neg()
    ):
        return None
    m, k = a.shape
    n = b.shape[1]
    if a.stride() != (1, m) or b.stride() != (n, 1):
        return None
    with torch_device_fn.device(a.device):
        choice = _plan(m, n, k, get_sm_count())
        if choice is None:
            return None
        return _execute(a, b, out, tf32, *choice)
