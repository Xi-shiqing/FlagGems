import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
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


def _allow_bmm_tf32(batch, M, N):
    """Avoid RNE conversion for multi-row, narrow-output BMMs on PPU.

    On the captured Protenix shape [16, 511, 511] x [16, 511, 24],
    converting both operands to scratch costs more than the strict-FP32 dot.
    The M=1, tiny-N Protenix case has a different launch/occupancy profile and
    remains eligible for TF32, as do wider-output BMMs.
    """
    return _allow_tf32() and not (M > 1 and N <= 32)


@triton.jit
def _bmm_tf32_inline_rne(
    A, B, O,
                           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           BATCH: tl.constexpr,
                           AS0: tl.constexpr, AS1: tl.constexpr, AS2: tl.constexpr,
                           BS0: tl.constexpr, BS1: tl.constexpr, BS2: tl.constexpr,
                           OS0: tl.constexpr, OS1: tl.constexpr, OS2: tl.constexpr,
                           TILE_M: tl.constexpr, TILE_N: tl.constexpr,
                           TILE_K: tl.constexpr):
    mi = tl.program_id(0) * TILE_M + tl.arange(0, TILE_M)
    ni = tl.program_id(1) * TILE_N + tl.arange(0, TILE_N)
    batch = tl.program_id(2)
    ki = tl.arange(0, TILE_K)
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for kb in range(tl.cdiv(K, TILE_K)):
        kk = kb * TILE_K + ki
        a = tl.load(A + batch * AS0 + mi[:, None] * AS1 + kk[None, :] * AS2,
                    (mi[:, None] < M) & (kk[None, :] < K), 0.0)
        b = tl.load(B + batch * BS0 + kk[:, None] * BS1 + ni[None, :] * BS2,
                    (kk[:, None] < K) & (ni[None, :] < N), 0.0)
        acc += tl.dot(round_tf32_rne(a, True), round_tf32_rne(b, True),
                      out_dtype=tl.float32, allow_tf32=True)
    tl.store(O + batch * OS0 + mi[:, None] * OS1 + ni[None, :] * OS2,
             acc, (mi[:, None] < M) & (ni[None, :] < N))


def _bmm_rne_once(A, B, out):
    batch, M, K = A.shape
    _, _, N = B.shape
    # For small operands, keep the single-kernel path to avoid an extra launch.
    if A.numel() + B.numel() < (1 << 20):
        return False
    # Native RNE is applied at the matched dot tile; no scratch tensors.
    tile_m = 16 if M < 16 else 32 if M < 256 else 64
    tile_n = 16 if N <= 16 else 32 if N < 256 else 64
    _bmm_tf32_inline_rne[(triton.cdiv(M, tile_m), triton.cdiv(N, tile_n), batch)](
        A, B, out, M, N, K, batch,
        *A.stride(), *B.stride(), *out.stride(),
        tile_m, tile_n, 32, num_warps=4,
    )
    return True


@libentry()
@libtuner(
    configs=[triton.Config(
        # Protenix real-shape candidate: identical to the production result on
        # all captured high-frequency shapes while reducing weighted BMM time.
        {"TILE_M": 64, "TILE_N": 32, "TILE_K": 32, "GROUP_M": 1},
        num_warps=4, num_stages=2,
    )],
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=runtime.get_expand_config("bmm")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else [
        "log",
        "log",
        "log",
        "align32",
        "align32",
    ],
)
@triton.heuristics(runtime.get_heuristic_config("bmm"))
@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    # batch offsets
    pid_b = tle.program_id(2)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    O += pid_b * stride_ob

    pidx = tle.program_id(0)
    pidy = tle.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = tle.num_programs(0)
        gridy = tle.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M

        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    o_ptrs = O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    num_iters = tl.cdiv(K, TILE_K)
    o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask_a)
        b = tl.load(b_ptrs, mask_b)

        offs_k += TILE_K
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

        o += tl.dot(
            round_tf32_rne(a, ALLOW_TF32),
            round_tf32_rne(b, ALLOW_TF32),
            allow_tf32=ALLOW_TF32,
        )

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M and not DIVISIBLE_N:
        mask_c = mask_n[None, :]
    elif not DIVISIBLE_M and DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, o, mask_c)


def bmm(A, B):
    logger.debug("GEMS BMM")
    assert A.shape[0] == B.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    batch, M, K = A.shape
    _, _, N = B.shape
    capture_real_operands(
        "bmm", [A, B], flops=2 * batch * M * N * K,
        output_elements=batch * M * N,
    )
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
    allow_tf32 = _allow_bmm_tf32(batch, M, N)

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        if allow_tf32 and A.dtype == torch.float32 and _bmm_rne_once(A, B, out):
            return out
        bmm_kernel[grid_fn](
            A,
            B,
            out,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            ALLOW_TF32=allow_tf32,
        )
    return out


def bmm_out(A, B, out):
    logger.debug("GEMS BMM_OUT")
    assert A.shape[0] == B.shape[0] == out.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    batch, M, K = A.shape
    _, _, N = B.shape
    capture_real_operands(
        "bmm", [A, B], flops=2 * batch * M * N * K,
        output_elements=batch * M * N,
    )
    allow_tf32 = _allow_bmm_tf32(batch, M, N)

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        if allow_tf32 and A.dtype == torch.float32 and _bmm_rne_once(A, B, out):
            return out
        bmm_kernel[grid_fn](
            A,
            B,
            out,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            ALLOW_TF32=allow_tf32,
        )
    return out
