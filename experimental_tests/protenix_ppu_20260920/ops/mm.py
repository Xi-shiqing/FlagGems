import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.mm_streamk import streamk_mm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.device_info import get_device_capability, get_sm_count

CACHE_USAGE_THRESHOLD = 0.8

logger = logging.getLogger(__name__)


def _allow_tf32():
    """Read Protenix's runtime TF32 switch at kernel launch time."""
    return bool(torch.backends.cuda.matmul.allow_tf32)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@libtuner(
    configs=[triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
        num_warps=4, num_stages=1,
    )],
    # Add 'stride_am' and 'stride_bk' to trigger autotune for tensors with the same shape but different strides.
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def mm_kernel_general(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    # matrix multiplication
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)
    prev_multiple = prev_multiple_of(K, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_multiple, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=ALLOW_TF32)

    # loop peeling
    rk = (prev_multiple + tl.arange(0, BLOCK_K)).to(tl.int64)
    mask_k = rk < K
    a = tl.load(
        A + (ram[:, None] * stride_am + rk[None, :] * stride_ak),
        mask=mask_k[None, :],
        other=0.0,
    )
    b = tl.load(
        B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn),
        mask=mask_k[:, None],
        other=0.0,
    )
    if a.dtype != b.dtype:
        a = a.to(C.dtype.element_ty)
        b = b.to(C.dtype.element_ty)
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=ALLOW_TF32)

    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


@libentry()
@triton.jit
def mm_kernel_h100_fp32_splitk(
    A, B, partials, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Deterministic FP32 split-K GEMM used for the PPU/H100 protocol."""
    pid = tle.program_id(0)
    split_id = tle.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    chunk_k = tl.cdiv(K, SPLIT_K)
    split_start = split_id * chunk_k
    split_end = tl.minimum(split_start + chunk_k, K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for offset_k in range(0, chunk_k, BLOCK_K):
        rk = split_start + offset_k + tl.arange(0, BLOCK_K)
        mask_k = rk < split_end
        a = tl.load(
            A + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm < M)[:, None] & mask_k[None, :], other=0.0,
        )
        b = tl.load(
            B + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=mask_k[:, None] & (rn < N)[None, :], other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=ALLOW_TF32)
    offset = (split_id * M + rm)[:, None] * N + rn[None, :]
    tl.store(
        partials + offset,
        acc,
        mask=(rm < M)[:, None] & (rn < N)[None, :],
    )


@libentry()
@triton.jit
def mm_kernel_h100_fp32_splitk_reduce(
    partials, C, elements, stride_cm, stride_cn, N,
    split_k, BLOCK: tl.constexpr,
):
    offsets = tle.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for split_id in range(0, split_k):
        acc += tl.load(
            partials + split_id * elements + offsets,
            mask=mask,
            other=0.0,
        )
    rows = offsets // N
    cols = offsets % N
    tl.store(C + rows * stride_cm + cols * stride_cn, acc, mask=mask)


def _ppu_h100_splitk_parts(M, N, K):
    # Shape-aware deterministic schedules selected from the H100 FP32 probe.
    # This is still a pure FlagGems kernel; no PPU-native fallback is involved.
    if K > 8192 or N > 512:
        return None
    if M <= 8:
        return 2
    if M <= 512:
        if K > 4096:
            return 32
        if K <= 256:
            return 2 if N <= 128 else 3
        return 8
    if K <= 256 and N >= 256:
        return 2
    return 3


def ppu_h100_fp32_splitk_scenario(a, b, M, N, K):
    return (
        os.environ.get("FLAG_GEMS_PPU_H100_SPLITK", "1") == "1"
        and os.path.isdir("/usr/local/PPU_SDK")
        and a.dtype is torch.float32
        and b.dtype is torch.float32
        and M > 0 and N > 0 and K > 0
        and _ppu_h100_splitk_parts(M, N, K) is not None
    )


def ppu_h100_fp32_splitk_mm(a, b, c, M, N, K):
    split_k = _ppu_h100_splitk_parts(M, N, K)
    assert split_k is not None
    block = 64
    partials = torch.empty(
        (split_k, M, N), device=a.device, dtype=torch.float32
    )
    grid = (triton.cdiv(M, block) * triton.cdiv(N, block), split_k)
    with torch_device_fn.device(a.device):
        mm_kernel_h100_fp32_splitk[grid](
            a, b, partials, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            SPLIT_K=split_k, BLOCK_M=block, BLOCK_N=block, BLOCK_K=32,
            ALLOW_TF32=_allow_tf32(),
            num_warps=4, num_stages=2,
        )
        mm_kernel_h100_fp32_splitk_reduce[(triton.cdiv(M * N, 256),)](
            partials, c, M * N, c.stride(0), c.stride(1), N,
            split_k, BLOCK=256, num_warps=4,
        )
    return c


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def general_mm(a, b, c, M, N, K):
    logger.debug(
        "GEMS MM, [mm scenario]: general, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s",
        M,
        N,
        K,
        a.stride(0) == 1,
        b.stride(0) == 1,
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_general[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            GROUP_M=8,
            ALLOW_TF32=_allow_tf32(),
        )
    return c


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_self_transpose"),
    key=["M", "K", "stride_am", "stride_ak"],
    strategy=["align32", "align32", "align32", "align32"],
    warmup=2,
    rep=4,
)
@triton.jit
def mm_kernel_syrk(
    A,
    C,
    M,
    K,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)

    # Packed lower-triangular launch domain:
    #   pid = row * (row + 1) / 2 + col, where 0 <= col <= row.
    #
    # Invert the triangular-number indexing by solving:
    #   row^2 + row - 2 * pid = 0
    # => row = (-1 + sqrt(1 + 8 * pid)) / 2
    #
    # We take floor(...) as the candidate row, then apply an integer +/-1 correction
    # because fp32 sqrt can be off near triangular-number boundaries.
    pid_f = pid.to(tl.float32)
    pid_m = tl.floor((tl.sqrt(8.0 * pid_f + 1.0) - 1.0) / 2.0).to(tl.int32)
    tri_start = pid_m * (pid_m + 1) // 2
    pid_m = tl.where(tri_start > pid, pid_m - 1, pid_m)
    next_tri_start = (pid_m + 1) * (pid_m + 2) // 2
    pid_m = tl.where(next_tri_start <= pid, pid_m + 1, pid_m)
    tri_start = pid_m * (pid_m + 1) // 2
    pid_n = pid - tri_start

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    ran = tl.max_contiguous(tl.multiple_of(rn % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)

    for start_k in range(0, K, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        mask_k = rk < K
        a = tl.load(
            A + (ram[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            A + (rk[:, None] * stride_ak + ran[None, :] * stride_am),
            mask=mask_k[:, None],
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=ALLOW_TF32)

    out = acc.to(C.dtype.element_ty)
    c_ptr = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < M)[None, :]
    tl.store(c_ptr, out, mask=mask)

    if pid_m > pid_n:
        c_t_ptr = C + (rn[:, None] * stride_cm + rm[None, :] * stride_cn)
        mask_t = (rn < M)[:, None] & (rm < M)[None, :]
        tl.store(c_t_ptr, tl.trans(out), mask=mask_t)


def is_syrk_transpose_pair(a, b):
    return (
        a.ndim == 2
        and b.ndim == 2
        and a.shape[0] == b.shape[1]
        and a.shape[1] == b.shape[0]
        and a.stride(0) == b.stride(1)
        and a.stride(1) == b.stride(0)
        and a.storage_offset() == b.storage_offset()
        and a.data_ptr() == b.data_ptr()
    )


def syrk_mm(a, c, M, K):
    grid = lambda META: (
        # Number of tile rows is tiles = ceil(M / BLOCK_M).
        # Packed lower triangle contains:
        #   1 + 2 + ... + tiles = tiles * (tiles + 1) / 2
        triton.cdiv(M, META["BLOCK_M"])
        * (triton.cdiv(M, META["BLOCK_M"]) + 1)
        // 2,
    )
    with torch_device_fn.device(a.device):
        mm_kernel_syrk[grid](
            a,
            c,
            M,
            K,
            a.stride(0),
            a.stride(1),
            c.stride(0),
            c.stride(1),
            ALLOW_TF32=_allow_tf32(),
        )
    return c


def streamk_scenario(a, b, M, N, K):
    # TODO: this my change sometime according to the realbenchmark result
    # Currently, the best configuration for streamk has only been tested on A100(capability[0] == 8).
    # The optimal settings for other devices need to be determined through real testing.
    capability = get_device_capability()
    return (
        capability[0] == 8
        and a.dtype in [torch.float16, torch.bfloat16]
        and b.dtype in [torch.float16, torch.bfloat16]
        and a.is_contiguous()
        and b.is_contiguous()
        and K > M * 5
        and K > N * 5
    )


@triton.jit
def mm_direct_fused(A, B, C, M, N, K,
                    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                    SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tle.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    rm = ((pid // grid_n) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = ((pid % grid_n) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    chunk_k = tl.cdiv(K, SPLIT_K)
    result = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split_id in range(0, SPLIT_K):
        split_start = split_id * chunk_k
        split_end = tl.minimum(split_start + chunk_k, K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for offset_k in range(0, chunk_k, BLOCK_K):
            rk = (split_start + offset_k + tl.arange(0, BLOCK_K)).to(tl.int64)
            mask_k = rk < split_end
            a = tl.load(A + rm[:, None] * stride_am + rk[None, :] * stride_ak,
                        mask=(rm < M)[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(B + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                        mask=mask_k[:, None] & (rn < N)[None, :], other=0.0)
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        if SPLIT_K == 1:
            result = acc
        else:
            result += acc
    tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             result, mask=(rm < M)[:, None] & (rn < N)[None, :])


# PPU inference: copy the RHS per call; never cache model weights.
_PPU_PACK_RHS_AVAILABLE = os.path.isdir("/usr/local/PPU_SDK")


def _should_pack_rhs(a, b):
    return (
        _PPU_PACK_RHS_AVAILABLE
        and not torch.is_grad_enabled()
        and a.dtype is torch.float32 and b.dtype is torch.float32
        and a.shape[0] >= 16384
        and 64 <= a.shape[1] <= 1024
        and 64 <= b.shape[1] <= 1024
        and a.shape[0] * a.shape[1] * b.shape[1] >= 536870912
        and a.device.type == "cuda"
        and a.is_contiguous()
        and b.stride(0) == 1 and b.stride(1) == b.shape[0]
        and not _allow_tf32()
    )


def _maybe_pack_rhs(a, b):
    return b.contiguous() if _should_pack_rhs(a, b) else b



_DIRECT_ENABLED = True


def _should_direct(a, b):
    return (
        _DIRECT_ENABLED and _PPU_PACK_RHS_AVAILABLE
        and not torch.is_grad_enabled()
        and a.dtype is torch.float32 and b.dtype is torch.float32
        and a.device.type == "cuda"
        and a.shape[0] > 0
        and 64 <= a.shape[1] <= 1024
        and 128 <= b.shape[1] <= 1024
        and ((b.shape[1] <= 512 and a.shape[1] <= 512
              and (a.shape[1] <= 256 or a.shape[0] >= 4096))
             or (b.shape[1] > 512 and a.shape[0] >= 4096))
        and a.is_contiguous()
        and b.stride(0) == 1 and b.stride(1) == b.shape[0]
        and os.environ.get("FLAG_GEMS_PPU_H100_SPLITK", "1") == "1"
        and not _allow_tf32()
    )


def _direct_pack_profitable(a, b):
    # Small launches prefer no copy; wide/large products amortize the conversion.
    return a.shape[0] >= 131072 or ((a.shape[1] >= 256 or b.shape[1] > 512) and a.shape[0] >= 16384)


def _direct_mm(a, b, c, M, N, K):
    # N>512 uses the original unsplit general-MM accumulation order.
    split_k = _ppu_h100_splitk_parts(M, N, K) or 1
    with torch_device_fn.device(a.device):
        mm_direct_fused[(triton.cdiv(M, 64) * triton.cdiv(N, 128),)](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            SPLIT_K=split_k, BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
            num_warps=4, num_stages=1,
        )
    return c


def _direct_out_safe(a, b, out):
    return (
        out.dtype is torch.float32 and out.device == a.device
        and tuple(out.shape) == (a.shape[0], b.shape[1])
        and (out.stride(0) >= out.shape[1] * out.stride(1) > 0
             or out.stride(1) >= out.shape[0] * out.stride(0) > 0)
        and out.untyped_storage().data_ptr() != a.untyped_storage().data_ptr()
        and out.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    )


def mm(a, b):
    device = a.device
    if is_syrk_transpose_pair(a, b):
        M, K = a.shape
        c = torch.empty((M, M), device=device, dtype=a.dtype)
        return syrk_mm(a, c, M, K)
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    use_direct = _should_direct(a, b)
    if use_direct and _direct_pack_profitable(a, b):
        b = b.contiguous()
    elif not use_direct and _DIRECT_ENABLED:
        b = _maybe_pack_rhs(a, b)
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    # l2_cache_size = get_l2_cache_size()
    sm_count = get_sm_count()
    if use_direct:
        return _direct_mm(a, b, c, M, N, K)
    if ppu_h100_fp32_splitk_scenario(a, b, M, N, K):
        return ppu_h100_fp32_splitk_mm(a, b, c, M, N, K)
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, c, M, N, K, sm_count=sm_count)
    else:
        return general_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    if is_syrk_transpose_pair(a, b):
        M, K = a.shape
        return syrk_mm(a, out, M, K)
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    use_direct = _should_direct(a, b)
    if use_direct and _direct_pack_profitable(a, b):
        b = b.contiguous()
    elif not use_direct and _DIRECT_ENABLED:
        b = _maybe_pack_rhs(a, b)
    # l2_cache_size = get_l2_cache_size()
    sm_count = get_sm_count()
    if use_direct and _direct_out_safe(a, b, out):
        return _direct_mm(a, b, out, M, N, K)
    if ppu_h100_fp32_splitk_scenario(a, b, M, N, K):
        return ppu_h100_fp32_splitk_mm(a, b, out, M, N, K)
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, out, M, N, K, sm_count=sm_count)
    else:
        return general_mm(a, b, out, M, N, K)
