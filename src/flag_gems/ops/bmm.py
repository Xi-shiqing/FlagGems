# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _use_packed_thead_rhs(A, B):
    """Whether copying the batch-interleaved RHS is cheaper than strided BMM."""
    batch, M, K = A.shape
    if (
        runtime.device.vendor_name != "thead"
        or A.dtype != torch.float32
        or B.dtype != torch.float32
    ):
        return False

    # The public Protenix trace contains a dominant pairformer contraction
    # with B=[batch,693,N] stored in either batch-interleaved permutation:
    # (1,N*batch,batch) or (1,batch,693*batch).  On PPU the generic strided
    # kernel is 40--49% slower than one contiguous copy plus BMM for these
    # exact shapes, so opt in to the measured packing path.  Keep the shape
    # guard narrow: copying arbitrary views can cost more than the kernel.
    if (
        batch in (64, 256)
        and M == 693
        and K == 693
        and os.getenv("FLAG_GEMS_PPU_BMM_HOTPACK", "1") == "1"
    ):
        return B.stride(0) == 1 and (
            (B.stride(1) == batch and B.stride(2) == K * batch)
            or (B.stride(1) == B.shape[2] * batch and B.stride(2) == batch)
        )

    # The broader, older batch-128 route remains explicitly opt-in.  It was
    # measured on a different trace and must not change unrelated workloads.
    if os.getenv("FLAG_GEMS_PPU_PACKED_BMM", "0") != "1":
        return False

    return (
        batch == 128
        and M >= 512
        and K >= 512
        and B.stride(0) == 1
        and B.stride(1) == batch
        and B.stride(2) == K * batch
    )


def _bmm_config_signature(config):
    meta = config.all_kwargs()
    return (
        int(meta["TILE_M"]),
        int(meta["TILE_N"]),
        int(meta["TILE_K"]),
        int(meta["GROUP_M"]),
        int(meta["num_warps"]),
        int(meta["num_stages"]),
    )


def _prune_thead_fp32_bmm_configs(configs, nargs, **kwargs):
    """Use measured tiles for Protenix's dominant PPU BMM layouts."""
    # Triton's Autotuner passes the positional tensors in ``nargs`` and the
    # scalar launch metadata in ``kwargs``.  Inspect both: when packed-BMM is
    # requested, the 64x64 tile is valid only for the same batch/stride
    # predicate that controls the actual ``B.contiguous()`` copy.  Otherwise
    # use the measured strided-RHS tile even if the environment variable is
    # enabled; applying the packed tile to an un-packed call made the first
    # end-to-end A/B arm slower.
    A = nargs.get("A")
    B = nargs.get("B")
    if A is None or B is None:
        return configs
    M = int(kwargs.get("M", nargs.get("M", 0)))
    K = int(kwargs.get("K", nargs.get("K", 0)))
    N = int(kwargs.get("N", nargs.get("N", 0)))
    # The pair-attention trace has one very frequent small-K shape where the
    # generic tuner leaves a little performance on the table.  Keep this
    # experiment opt-in until the full-model A/B confirms it: unlike the
    # broad rules below, it must match the complete shape and strides.
    exact_hotshape = (
        runtime.device.vendor_name == "thead"
        and getattr(A, "dtype", None) == torch.float32
        and getattr(B, "dtype", None) == torch.float32
        and tuple(A.shape) == (5544, 693, 32)
        and tuple(B.shape) == (5544, 32, 693)
        and tuple(A.stride()) == (22176, 32, 1)
        and tuple(B.stride()) == (22176, 693, 1)
    )
    # This exact shape is the dominant small-K pairformer contraction.  The
    # measured 64x64x32/W8/S2 schedule is now the default for this one layout;
    # FLAG_GEMS_PPU_BMM_HOTSHAPE=0 remains an explicit rollback switch.
    if (
        os.getenv("FLAG_GEMS_PPU_BMM_HOTSHAPE", "1" if exact_hotshape else "0")
        == "1"
        and exact_hotshape
    ):
        target = (64, 64, 32, 1, 8, 2)
        selected = [config for config in configs if _bmm_config_signature(config) == target]
        if len(selected) == 1:
            return selected
        # This exploratory schedule is intentionally not in the shared YAML
        # table yet; construct it locally so the opt-in path remains isolated.
        return [
            triton.Config(
                {"TILE_M": 64, "TILE_N": 64, "TILE_K": 32, "GROUP_M": 1},
                num_warps=8,
                num_stages=2,
            )
        ]
    if (
        runtime.device.vendor_name != "thead"
        or getattr(A, "dtype", None) != torch.float32
        or getattr(B, "dtype", None) != torch.float32
        or M < 512
        or K < 512
        or N < 64
    ):
        return configs
    packed = (
        os.getenv("FLAG_GEMS_PPU_PACKED_BMM", "0") == "1"
        and int(A.shape[0]) == 128
        and tuple(B.stride()) == (1, int(A.shape[0]), K * int(A.shape[0]))
    )
    if packed:
        # Packed RHS: a 64x64 tile minimizes the PPU's batch-stride penalty.
        target = (64, 64, 32, 2, 4, 3)
    elif int(A.shape[0]) <= 8 and N >= 2048:
        # The small-batch, very-wide projection observed in the diffusion
        # trace is occupancy-bound; M64 avoids launching mostly empty M tiles.
        target = (64, 64, 32, 2, 4, 3)
    elif int(A.shape[0]) == 1 and M >= 10_000 and K >= 1_024:
        # The atom projection has a long M tail and a narrow N tile.  The
        # exhaustive real-shape search selected two pipeline stages here.
        target = (128, 32, 32, 2, 4, 2)
    elif N <= 128:
        # Narrow pairformer projections waste a 64-wide N tile.
        target = (128, 32, 32, 2, 4, 3)
    else:
        # Dominant pairformer projections with interleaved RHS.
        target = (128, 64, 32, 2, 4, 3)
    selected = [config for config in configs if _bmm_config_signature(config) == target]
    return selected if len(selected) == 1 else configs


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bmm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=[
        "log",
        "log",
        "log",
        "align32",
        "align32",
    ],
    prune_configs_by={"early_config_prune": _prune_thead_fp32_bmm_configs},
    flagtune_op_name="bmm",
    flagtune_expand_op_name="bmm",
    flagtune_pre_hook=None,
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
    IS_FP64: tl.constexpr = False,
):
    # batch offsets
    pid_b = ext.program_id(2)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    O += pid_b * stride_ob

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
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
    if IS_FP64:
        o = tl.zeros((TILE_M, TILE_N), dtype=tl.float64)
    else:
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

        o += tl.dot(a, b, allow_tf32=False)

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
    if _use_packed_thead_rhs(A, B):
        B = B.contiguous()
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
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
            IS_FP64=A.dtype == torch.float64,
        )
    return out


def bmm_out(A, B, out):
    logger.debug("GEMS BMM_OUT")
    assert A.shape[0] == B.shape[0] == out.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    batch, M, K = A.shape
    _, _, N = B.shape
    if _use_packed_thead_rhs(A, B):
        B = B.contiguous()

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
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
            IS_FP64=A.dtype == torch.float64,
        )
    return out
