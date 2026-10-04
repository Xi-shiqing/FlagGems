import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.ops.tf32_precision import (
    capture_real_operands,
    round_tf32_rne,
    tf32_enabled_for_active_module,
)

logger = logging.getLogger(__name__)


def _allow_tf32():
    """Read the global TF32 switch and the per-module restore mask."""
    return tf32_enabled_for_active_module()


def _ppu_h100_splitk_parts(M, N, K):
    # Shape-aware deterministic schedules selected from the H100 FP32 probe.
    # This remains a pure FlagGems kernel; no PPU-native fallback is involved.
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


@libentry()
@triton.jit
def addmm_kernel_h100_fp32_splitk(
    A, B, bias, C, alpha, beta, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_im, stride_in, stride_cm, stride_cn,
    SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
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
        acc += tl.dot(
            round_tf32_rne(a, ALLOW_TF32),
            round_tf32_rne(b, ALLOW_TF32),
            out_dtype=tl.float32,
            allow_tf32=ALLOW_TF32,
        )
    offset = (split_id * M + rm)[:, None] * N + rn[None, :]
    tl.store(
        bias + offset,
        acc,
        mask=(rm < M)[:, None] & (rn < N)[None, :],
    )


@libentry()
@triton.jit
def addmm_kernel_h100_fp32_splitk_reduce(
    partials, bias, C, elements, alpha, beta,
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
    x = tl.load(bias + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, acc * alpha + x * beta, mask=mask)


def ppu_h100_fp32_splitk_addmm(bias, mat1, mat2, out, alpha, beta, M, N, K):
    split_k = _ppu_h100_splitk_parts(M, N, K)
    assert split_k is not None
    bias = bias.contiguous()
    block = 64
    partials = torch.empty(
        (split_k, M, N), device=mat1.device, dtype=torch.float32
    )
    grid = (triton.cdiv(M, block) * triton.cdiv(N, block), split_k)
    with torch_device_fn.device(mat1.device):
        addmm_kernel_h100_fp32_splitk[grid](
            mat1, mat2, partials, out, alpha, beta, M, N, K,
            mat1.stride(0), mat1.stride(1), mat2.stride(0), mat2.stride(1),
            bias.stride(0), bias.stride(1), out.stride(0), out.stride(1),
            SPLIT_K=split_k, BLOCK_M=block, BLOCK_N=block, BLOCK_K=32,
            ALLOW_TF32=_allow_tf32(),
            num_warps=4, num_stages=2,
        )
        addmm_kernel_h100_fp32_splitk_reduce[(triton.cdiv(M * N, 256),)](
            partials, bias, out, M * N, alpha, beta,
            split_k, BLOCK=256, num_warps=4,
        )
    return out


@libentry()
@libtuner(
    configs=[triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
        num_warps=4, num_stages=1,
    )],
    key=["M", "N", "K"],
    strategy=runtime.get_expand_config("addmm")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else ["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(
            round_tf32_rne(a, ALLOW_TF32),
            round_tf32_rne(b, ALLOW_TF32),
            allow_tf32=ALLOW_TF32,
        )
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
    bias = tl.load(i_ptrs, mask=c_mask, other=0.0)

    accumulator = accumulator * alpha + bias * beta
    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    capture_real_operands(
        "addmm",
        [bias, mat1, mat2],
        flops=2 * int(mat1.shape[0]) * int(mat2.shape[1]) * int(mat1.shape[1]),
        output_elements=int(mat1.shape[0]) * int(mat2.shape[1]),
        alpha=alpha,
        beta=beta,
    )
    M, K = mat1.shape
    _, N = mat2.shape
    logger.debug(
        "GEMS ADDMM, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s, [bias column-major]: %s",
        M,
        N,
        K,
        mat1.stride(0) == 1,
        mat2.stride(0) == 1,
        bias.stride(0) == 1,
    )
    mat1 = mat1.contiguous()
    # mat2 = mat2.contiguous()
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    bias = bias.broadcast_to(out.shape)
    if (
        os.path.isdir("/usr/local/PPU_SDK")
        and os.environ.get("FLAG_GEMS_PPU_H100_SPLITK", "1") == "1"
        and mat1.dtype is torch.float32 and mat2.dtype is torch.float32
        and _ppu_h100_splitk_parts(M, N, K) is not None
    ):
        return ppu_h100_fp32_splitk_addmm(
            bias, mat1, mat2, out, alpha, beta, M, N, K
        )

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(mat1.device):
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            bias.stride(0),
            bias.stride(1),
            out.stride(0),
            out.stride(1),
            ALLOW_TF32=_allow_tf32(),
        )
    return out


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    capture_real_operands(
        "addmm",
        [bias, mat1, mat2],
        flops=2 * int(mat1.shape[0]) * int(mat2.shape[1]) * int(mat1.shape[1]),
        output_elements=int(mat1.shape[0]) * int(mat2.shape[1]),
        alpha=alpha,
        beta=beta,
    )
    M, K = mat1.shape
    _, N = mat2.shape
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    logger.debug(
        "GEMS ADDMM_OUT, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s, [bias column-major]: %s",
        M,
        N,
        K,
        mat1.stride(0) == 1,
        mat2.stride(0) == 1,
        bias.stride(0) == 1,
    )
    mat1 = mat1.contiguous()
    bias = bias.broadcast_to(out.shape)
    if (
        os.path.isdir("/usr/local/PPU_SDK")
        and os.environ.get("FLAG_GEMS_PPU_H100_SPLITK", "1") == "1"
        and mat1.dtype is torch.float32 and mat2.dtype is torch.float32
        and _ppu_h100_splitk_parts(M, N, K) is not None
    ):
        return ppu_h100_fp32_splitk_addmm(
            bias, mat1, mat2, out, alpha, beta, M, N, K
        )

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(mat1.device):
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            bias.stride(0),
            bias.stride(1),
            out.stride(0),
            out.stride(1),
            ALLOW_TF32=_allow_tf32(),
        )
    return out
