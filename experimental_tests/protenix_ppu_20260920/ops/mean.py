import logging
import math
from functools import reduce

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def mean_dim_kernel_h100_inner_fp32(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
):
    """FP32 contiguous-last reduction with the H100 CUDA 2.7 tree.

    H100's vectorized reduction uses four serial accumulators per lane and an
    increasing-offset warp tree.  PPU's ``tl.sum`` uses the opposite tree;
    reversing the five lane bits before loading makes its result identical.
    """
    pid_m = tle.program_id(0)
    lane = tl.arange(0, 32)
    rev = (
        ((lane & 1) << 4)
        | ((lane & 2) << 2)
        | (lane & 4)
        | ((lane & 8) >> 2)
        | ((lane & 16) >> 4)
    )
    acc0 = tl.zeros((32,), dtype=tl.float32)
    acc1 = tl.zeros((32,), dtype=tl.float32)
    acc2 = tl.zeros((32,), dtype=tl.float32)
    acc3 = tl.zeros((32,), dtype=tl.float32)
    base = pid_m * N
    # Reduce.cuh first aligns each row independently.  A row starts at
    # ``pid_m*N`` elements, so rows can have different four-element head/tail
    # handling even when the tensor itself is contiguous.
    row_shift = (pid_m * N) % 4
    data_shift = (4 - row_shift) % 4
    data_base = base + data_shift
    end = N - data_shift
    head_x = rev
    head = (data_shift > 0) & (head_x >= row_shift) & (head_x < 4)
    acc0 = acc0 + tl.load(input_ptr + base + head_x - row_shift, mask=head, other=0.0)

    # Only complete four-element vectors belong to the CUDA vector loop.
    for start in range(0, N, 128):
        offset = data_base + start + rev * 4
        valid = start + rev * 4 + 3 < end
        acc0 = acc0 + tl.load(input_ptr + offset + 0, mask=valid, other=0.0).to(tl.float32)
        acc1 = acc1 + tl.load(input_ptr + offset + 1, mask=valid, other=0.0).to(tl.float32)
        acc2 = acc2 + tl.load(input_ptr + offset + 2, mask=valid, other=0.0).to(tl.float32)
        acc3 = acc3 + tl.load(input_ptr + offset + 3, mask=valid, other=0.0).to(tl.float32)

    # CUDA's vectorized tail is accumulated into value_list[0] by physical
    # lanes 0..(tail_size-1).  ``rev`` maps those lanes back to the PPU tree.
    tail_start = end - (end % 4)
    tail = tail_start + rev
    acc0 = acc0 + tl.load(input_ptr + data_base + tail, mask=tail < end, other=0.0).to(tl.float32)
    lane_sum = acc0 + acc1
    lane_sum = lane_sum + acc2
    lane_sum = lane_sum + acc3
    total = tl.sum(lane_sum, axis=0)
    tl.store(output_ptr + pid_m, total * (1.0 / N))


@libentry()
@triton.jit
def mean_dim_kernel_h100_inner_small_fp32(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    """FP32 contiguous-last reduction for H100's non-vectorized path."""
    pid_m = tle.program_id(0)
    lane = tl.arange(0, BLOCK_X)
    if tl.constexpr(BLOCK_X == 1):
        rev = lane
    elif tl.constexpr(BLOCK_X == 2):
        rev = ((lane & 1) << 0) | ((lane & 0) >> 1)
    elif tl.constexpr(BLOCK_X == 4):
        rev = ((lane & 1) << 1) | ((lane & 2) >> 1)
    elif tl.constexpr(BLOCK_X == 8):
        rev = (
            ((lane & 1) << 2)
            | ((lane & 2) << 0)
            | ((lane & 4) >> 2)
        )
    elif tl.constexpr(BLOCK_X == 16):
        rev = (
            ((lane & 1) << 3)
            | ((lane & 2) << 1)
            | ((lane & 4) >> 1)
            | ((lane & 8) >> 3)
        )
    else:
        rev = (
            ((lane & 1) << 4)
            | ((lane & 2) << 2)
            | (lane & 4)
            | ((lane & 8) >> 2)
            | ((lane & 16) >> 4)
        )
    base = pid_m * N
    acc0 = tl.load(input_ptr + base + rev, mask=rev < N, other=0.0).to(tl.float32)
    acc1 = tl.load(input_ptr + base + rev + BLOCK_X, mask=rev + BLOCK_X < N, other=0.0).to(tl.float32)
    acc2 = tl.load(input_ptr + base + rev + 2 * BLOCK_X, mask=rev + 2 * BLOCK_X < N, other=0.0).to(tl.float32)
    acc3 = tl.load(input_ptr + base + rev + 3 * BLOCK_X, mask=rev + 3 * BLOCK_X < N, other=0.0).to(tl.float32)
    lane_sum = acc0 + acc1
    lane_sum = lane_sum + acc2
    lane_sum = lane_sum + acc3
    total = tl.sum(lane_sum, axis=0)
    tl.store(output_ptr + pid_m, total * (1.0 / N))


@libentry()
@triton.jit
def mean_dim_kernel_h100_inner_wide_vec_fp32(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    """H100 vectorized inner reduction for 64/128 logical lanes.

    PPU exposes a different cross-warp reduction tree.  We therefore keep
    one physical 32-lane warp, reproduce each logical CUDA warp's four-list
    thread reduction, and perform the CUDA shared stages explicitly before
    the already-correct 32-lane tree.
    """
    pid_m = tle.program_id(0)
    q = tl.arange(0, 32)
    rev = (
        ((q & 1) << 4)
        | ((q & 2) << 2)
        | (q & 4)
        | ((q & 8) >> 2)
        | ((q & 16) >> 4)
    )
    base = pid_m * N
    row_shift = (pid_m * N) % 4
    data_shift = (4 - row_shift) % 4
    data_base = base + data_shift
    end = N - data_shift
    if tl.constexpr(BLOCK_X == 64):
        x0 = rev
        x1 = 32 + rev
        a00 = tl.zeros((32,), dtype=tl.float32)
        a01 = tl.zeros((32,), dtype=tl.float32)
        a02 = tl.zeros((32,), dtype=tl.float32)
        a03 = tl.zeros((32,), dtype=tl.float32)
        a10 = tl.zeros((32,), dtype=tl.float32)
        a11 = tl.zeros((32,), dtype=tl.float32)
        a12 = tl.zeros((32,), dtype=tl.float32)
        a13 = tl.zeros((32,), dtype=tl.float32)
        head = (data_shift > 0) & (x0 >= row_shift) & (x0 < 4)
        a00 = a00 + tl.load(input_ptr + base + x0 - row_shift, mask=head, other=0.0)
        for start in range(0, N, BLOCK_X * 4):
            off0 = data_base + start + x0 * 4
            ok0 = start + x0 * 4 + 3 < end
            a00 = a00 + tl.load(input_ptr + off0 + 0, mask=ok0, other=0.0)
            a01 = a01 + tl.load(input_ptr + off0 + 1, mask=ok0, other=0.0)
            a02 = a02 + tl.load(input_ptr + off0 + 2, mask=ok0, other=0.0)
            a03 = a03 + tl.load(input_ptr + off0 + 3, mask=ok0, other=0.0)
            off1 = data_base + start + x1 * 4
            ok1 = start + x1 * 4 + 3 < end
            a10 = a10 + tl.load(input_ptr + off1 + 0, mask=ok1, other=0.0)
            a11 = a11 + tl.load(input_ptr + off1 + 1, mask=ok1, other=0.0)
            a12 = a12 + tl.load(input_ptr + off1 + 2, mask=ok1, other=0.0)
            a13 = a13 + tl.load(input_ptr + off1 + 3, mask=ok1, other=0.0)
        tail_start = end - (end % 4)
        a00 = a00 + tl.load(input_ptr + data_base + tail_start + x0, mask=tail_start + x0 < end, other=0.0)
        a10 = a10 + tl.load(input_ptr + data_base + tail_start + x1, mask=tail_start + x1 < end, other=0.0)
        z0 = a00 + a01
        z0 = z0 + a02
        z0 = z0 + a03
        z1 = a10 + a11
        z1 = z1 + a12
        z1 = z1 + a13
        # CUDA block_x_reduce first uses offset 32.
        lane_sum = z0 + z1
    else:
        x0 = rev
        x1 = 32 + rev
        x2 = 64 + rev
        x3 = 96 + rev
        a00 = tl.zeros((32,), dtype=tl.float32)
        a01 = tl.zeros((32,), dtype=tl.float32)
        a02 = tl.zeros((32,), dtype=tl.float32)
        a03 = tl.zeros((32,), dtype=tl.float32)
        a10 = tl.zeros((32,), dtype=tl.float32)
        a11 = tl.zeros((32,), dtype=tl.float32)
        a12 = tl.zeros((32,), dtype=tl.float32)
        a13 = tl.zeros((32,), dtype=tl.float32)
        a20 = tl.zeros((32,), dtype=tl.float32)
        a21 = tl.zeros((32,), dtype=tl.float32)
        a22 = tl.zeros((32,), dtype=tl.float32)
        a23 = tl.zeros((32,), dtype=tl.float32)
        a30 = tl.zeros((32,), dtype=tl.float32)
        a31 = tl.zeros((32,), dtype=tl.float32)
        a32 = tl.zeros((32,), dtype=tl.float32)
        a33 = tl.zeros((32,), dtype=tl.float32)
        head = (data_shift > 0) & (x0 >= row_shift) & (x0 < 4)
        a00 = a00 + tl.load(input_ptr + base + x0 - row_shift, mask=head, other=0.0)
        for start in range(0, N, BLOCK_X * 4):
            off0 = data_base + start + x0 * 4
            ok0 = start + x0 * 4 + 3 < end
            a00 = a00 + tl.load(input_ptr + off0 + 0, mask=ok0, other=0.0)
            a01 = a01 + tl.load(input_ptr + off0 + 1, mask=ok0, other=0.0)
            a02 = a02 + tl.load(input_ptr + off0 + 2, mask=ok0, other=0.0)
            a03 = a03 + tl.load(input_ptr + off0 + 3, mask=ok0, other=0.0)
            off1 = data_base + start + x1 * 4
            ok1 = start + x1 * 4 + 3 < end
            a10 = a10 + tl.load(input_ptr + off1 + 0, mask=ok1, other=0.0)
            a11 = a11 + tl.load(input_ptr + off1 + 1, mask=ok1, other=0.0)
            a12 = a12 + tl.load(input_ptr + off1 + 2, mask=ok1, other=0.0)
            a13 = a13 + tl.load(input_ptr + off1 + 3, mask=ok1, other=0.0)
            off2 = data_base + start + x2 * 4
            ok2 = start + x2 * 4 + 3 < end
            a20 = a20 + tl.load(input_ptr + off2 + 0, mask=ok2, other=0.0)
            a21 = a21 + tl.load(input_ptr + off2 + 1, mask=ok2, other=0.0)
            a22 = a22 + tl.load(input_ptr + off2 + 2, mask=ok2, other=0.0)
            a23 = a23 + tl.load(input_ptr + off2 + 3, mask=ok2, other=0.0)
            off3 = data_base + start + x3 * 4
            ok3 = start + x3 * 4 + 3 < end
            a30 = a30 + tl.load(input_ptr + off3 + 0, mask=ok3, other=0.0)
            a31 = a31 + tl.load(input_ptr + off3 + 1, mask=ok3, other=0.0)
            a32 = a32 + tl.load(input_ptr + off3 + 2, mask=ok3, other=0.0)
            a33 = a33 + tl.load(input_ptr + off3 + 3, mask=ok3, other=0.0)
        tail_start = end - (end % 4)
        a00 = a00 + tl.load(input_ptr + data_base + tail_start + x0, mask=tail_start + x0 < end, other=0.0)
        a10 = a10 + tl.load(input_ptr + data_base + tail_start + x1, mask=tail_start + x1 < end, other=0.0)
        a20 = a20 + tl.load(input_ptr + data_base + tail_start + x2, mask=tail_start + x2 < end, other=0.0)
        a30 = a30 + tl.load(input_ptr + data_base + tail_start + x3, mask=tail_start + x3 < end, other=0.0)
        z0 = a00 + a01
        z0 = z0 + a02
        z0 = z0 + a03
        z1 = a10 + a11
        z1 = z1 + a12
        z1 = z1 + a13
        z2 = a20 + a21
        z2 = z2 + a22
        z2 = z2 + a23
        z3 = a30 + a31
        z3 = z3 + a32
        z3 = z3 + a33
        # CUDA shared stages: offset 64, then offset 32.
        pair0 = z0 + z2
        pair1 = z1 + z3
        lane_sum = pair0 + pair1
    total = tl.sum(lane_sum, axis=0)
    tl.store(output_ptr + pid_m, total * (1.0 / N))


@libentry()
@triton.jit
def mean_dim_kernel_h100_inner_wide_partial_fp32(
    partial_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    BLOCK_X: tl.constexpr,
    INPUT_SHIFT: tl.constexpr,
):
    """One logical warp of the 256/512-lane H100 vectorized reduction."""
    pid_m = tle.program_id(0)
    pid_w = tle.program_id(1)
    q = tl.arange(0, 32)
    rev = (
        ((q & 1) << 4)
        | ((q & 2) << 2)
        | (q & 4)
        | ((q & 8) >> 2)
        | ((q & 16) >> 4)
    )
    x = pid_w * 32 + rev
    base = pid_m * N
    row_shift = (INPUT_SHIFT + base) % 4
    data_shift = (4 - row_shift) % 4
    data_base = base + data_shift
    end = N - data_shift
    a0 = tl.zeros((32,), dtype=tl.float32)
    a1 = tl.zeros((32,), dtype=tl.float32)
    a2 = tl.zeros((32,), dtype=tl.float32)
    a3 = tl.zeros((32,), dtype=tl.float32)
    head = (data_shift > 0) & (x >= row_shift) & (x < 4)
    a0 = a0 + tl.load(input_ptr + base + x - row_shift, mask=head, other=0.0)
    for start in range(0, N, BLOCK_X * 4):
        off = data_base + start + x * 4
        valid = start + x * 4 + 3 < end
        a0 = a0 + tl.load(input_ptr + off + 0, mask=valid, other=0.0)
        a1 = a1 + tl.load(input_ptr + off + 1, mask=valid, other=0.0)
        a2 = a2 + tl.load(input_ptr + off + 2, mask=valid, other=0.0)
        a3 = a3 + tl.load(input_ptr + off + 3, mask=valid, other=0.0)
    # H100 vectorized tail is assigned to value_list[0] by physical lanes.
    tail_start = end - (end % 4)
    a0 = a0 + tl.load(
        input_ptr + data_base + tail_start + x,
        mask=tail_start + x < end,
        other=0.0,
    )
    z = a0 + a1
    z = z + a2
    z = z + a3
    tl.store(partial_ptr + (pid_m * (BLOCK_X // 32) + pid_w) * 32 + q, z)


@libentry()
@triton.jit
def mean_dim_kernel_h100_inner_wide_combine_fp32(
    output_ptr,
    partial_ptr,
    M,
    N: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    """H100 offset tree for eight or sixteen stored logical warp partials."""
    pid_m = tle.program_id(0)
    q = tl.arange(0, 32)
    base = (pid_m * (BLOCK_X // 32)) * 32 + q
    z0 = tl.load(partial_ptr + base + 0 * 32)
    z1 = tl.load(partial_ptr + base + 1 * 32)
    z2 = tl.load(partial_ptr + base + 2 * 32)
    z3 = tl.load(partial_ptr + base + 3 * 32)
    z4 = tl.load(partial_ptr + base + 4 * 32)
    z5 = tl.load(partial_ptr + base + 5 * 32)
    z6 = tl.load(partial_ptr + base + 6 * 32)
    z7 = tl.load(partial_ptr + base + 7 * 32)
    if tl.constexpr(BLOCK_X == 512):
        z8 = tl.load(partial_ptr + base + 8 * 32)
        z9 = tl.load(partial_ptr + base + 9 * 32)
        z10 = tl.load(partial_ptr + base + 10 * 32)
        z11 = tl.load(partial_ptr + base + 11 * 32)
        z12 = tl.load(partial_ptr + base + 12 * 32)
        z13 = tl.load(partial_ptr + base + 13 * 32)
        z14 = tl.load(partial_ptr + base + 14 * 32)
        z15 = tl.load(partial_ptr + base + 15 * 32)
        # CUDA block_x_reduce: offsets 256, 128, 64, 32.
        p0 = z0 + z8
        p1 = z1 + z9
        p2 = z2 + z10
        p3 = z3 + z11
        p4 = z4 + z12
        p5 = z5 + z13
        p6 = z6 + z14
        p7 = z7 + z15
    else:
        p0 = z0
        p1 = z1
        p2 = z2
        p3 = z3
        p4 = z4
        p5 = z5
        p6 = z6
        p7 = z7
    r0 = p0 + p4
    r1 = p1 + p5
    r2 = p2 + p6
    r3 = p3 + p7
    s0 = r0 + r2
    s1 = r1 + r3
    lane_sum = s0 + s1
    total = tl.sum(lane_sum, axis=0)
    tl.store(output_ptr + pid_m, total * (1.0 / N))


@libentry()
@triton.jit
def mean_dim_kernel_h100_inner_wide_scalar_fp32(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    """H100 non-vectorized inner reduction for a 64/128-lane block."""
    pid_m = tle.program_id(0)
    q = tl.arange(0, 32)
    rev = (
        ((q & 1) << 4)
        | ((q & 2) << 2)
        | (q & 4)
        | ((q & 8) >> 2)
        | ((q & 16) >> 4)
    )
    base = pid_m * N
    if tl.constexpr(BLOCK_X == 64):
        x0 = rev
        x1 = 32 + rev
        a00 = tl.zeros((32,), dtype=tl.float32)
        a01 = tl.zeros((32,), dtype=tl.float32)
        a02 = tl.zeros((32,), dtype=tl.float32)
        a03 = tl.zeros((32,), dtype=tl.float32)
        a10 = tl.zeros((32,), dtype=tl.float32)
        a11 = tl.zeros((32,), dtype=tl.float32)
        a12 = tl.zeros((32,), dtype=tl.float32)
        a13 = tl.zeros((32,), dtype=tl.float32)
        for start in range(0, N, BLOCK_X * 4):
            off0 = base + start + x0
            off1 = off0 + BLOCK_X
            off2 = off1 + BLOCK_X
            off3 = off2 + BLOCK_X
            a00 = a00 + tl.load(input_ptr + off0, mask=off0 < base + N, other=0.0)
            a01 = a01 + tl.load(input_ptr + off1, mask=off1 < base + N, other=0.0)
            a02 = a02 + tl.load(input_ptr + off2, mask=off2 < base + N, other=0.0)
            a03 = a03 + tl.load(input_ptr + off3, mask=off3 < base + N, other=0.0)
            off0 = base + start + x1
            off1 = off0 + BLOCK_X
            off2 = off1 + BLOCK_X
            off3 = off2 + BLOCK_X
            a10 = a10 + tl.load(input_ptr + off0, mask=off0 < base + N, other=0.0)
            a11 = a11 + tl.load(input_ptr + off1, mask=off1 < base + N, other=0.0)
            a12 = a12 + tl.load(input_ptr + off2, mask=off2 < base + N, other=0.0)
            a13 = a13 + tl.load(input_ptr + off3, mask=off3 < base + N, other=0.0)
        z0 = a00 + a01
        z0 = z0 + a02
        z0 = z0 + a03
        z1 = a10 + a11
        z1 = z1 + a12
        z1 = z1 + a13
        lane_sum = z0 + z1
    else:
        x0 = rev
        x1 = 32 + rev
        x2 = 64 + rev
        x3 = 96 + rev
        a00 = tl.zeros((32,), dtype=tl.float32)
        a01 = tl.zeros((32,), dtype=tl.float32)
        a02 = tl.zeros((32,), dtype=tl.float32)
        a03 = tl.zeros((32,), dtype=tl.float32)
        a10 = tl.zeros((32,), dtype=tl.float32)
        a11 = tl.zeros((32,), dtype=tl.float32)
        a12 = tl.zeros((32,), dtype=tl.float32)
        a13 = tl.zeros((32,), dtype=tl.float32)
        a20 = tl.zeros((32,), dtype=tl.float32)
        a21 = tl.zeros((32,), dtype=tl.float32)
        a22 = tl.zeros((32,), dtype=tl.float32)
        a23 = tl.zeros((32,), dtype=tl.float32)
        a30 = tl.zeros((32,), dtype=tl.float32)
        a31 = tl.zeros((32,), dtype=tl.float32)
        a32 = tl.zeros((32,), dtype=tl.float32)
        a33 = tl.zeros((32,), dtype=tl.float32)
        for start in range(0, N, BLOCK_X * 4):
            off0 = base + start + x0
            off1 = off0 + BLOCK_X
            off2 = off1 + BLOCK_X
            off3 = off2 + BLOCK_X
            a00 = a00 + tl.load(input_ptr + off0, mask=off0 < base + N, other=0.0)
            a01 = a01 + tl.load(input_ptr + off1, mask=off1 < base + N, other=0.0)
            a02 = a02 + tl.load(input_ptr + off2, mask=off2 < base + N, other=0.0)
            a03 = a03 + tl.load(input_ptr + off3, mask=off3 < base + N, other=0.0)
            off0 = base + start + x1
            off1 = off0 + BLOCK_X
            off2 = off1 + BLOCK_X
            off3 = off2 + BLOCK_X
            a10 = a10 + tl.load(input_ptr + off0, mask=off0 < base + N, other=0.0)
            a11 = a11 + tl.load(input_ptr + off1, mask=off1 < base + N, other=0.0)
            a12 = a12 + tl.load(input_ptr + off2, mask=off2 < base + N, other=0.0)
            a13 = a13 + tl.load(input_ptr + off3, mask=off3 < base + N, other=0.0)
            off0 = base + start + x2
            off1 = off0 + BLOCK_X
            off2 = off1 + BLOCK_X
            off3 = off2 + BLOCK_X
            a20 = a20 + tl.load(input_ptr + off0, mask=off0 < base + N, other=0.0)
            a21 = a21 + tl.load(input_ptr + off1, mask=off1 < base + N, other=0.0)
            a22 = a22 + tl.load(input_ptr + off2, mask=off2 < base + N, other=0.0)
            a23 = a23 + tl.load(input_ptr + off3, mask=off3 < base + N, other=0.0)
            off0 = base + start + x3
            off1 = off0 + BLOCK_X
            off2 = off1 + BLOCK_X
            off3 = off2 + BLOCK_X
            a30 = a30 + tl.load(input_ptr + off0, mask=off0 < base + N, other=0.0)
            a31 = a31 + tl.load(input_ptr + off1, mask=off1 < base + N, other=0.0)
            a32 = a32 + tl.load(input_ptr + off2, mask=off2 < base + N, other=0.0)
            a33 = a33 + tl.load(input_ptr + off3, mask=off3 < base + N, other=0.0)
        z0 = a00 + a01
        z0 = z0 + a02
        z0 = z0 + a03
        z1 = a10 + a11
        z1 = z1 + a12
        z1 = z1 + a13
        z2 = a20 + a21
        z2 = z2 + a22
        z2 = z2 + a23
        z3 = a30 + a31
        z3 = z3 + a32
        z3 = z3 + a33
        pair0 = z0 + z2
        pair1 = z1 + z3
        lane_sum = pair0 + pair1
    total = tl.sum(lane_sum, axis=0)
    tl.store(output_ptr + pid_m, total * (1.0 / N))


@libentry()
@triton.jit
def mean_dim_kernel_h100_non_inner_fp32(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """FP32 ``[M, N, K].mean(dim=1)`` matching H100 block-y reduction.

    The CUDA reduction keeps four serial accumulators in each reduction lane,
    then combines lanes in a decreasing-offset tree.  PPU's reduction is
    hierarchical (within warp, then across warps); transposing the lane index
    converts that hierarchy to the flat CUDA order.
    """
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    y = tl.arange(0, BLOCK_Y)
    x = tl.arange(0, 1)
    warp_count = BLOCK_Y // 32
    reduction_y = (y % 32) * warp_count + y // 32
    acc0 = tl.zeros((BLOCK_Y, 1), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_Y, 1), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_Y, 1), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_Y, 1), dtype=tl.float32)
    base = pid_m * N * K
    for start in range(0, N, BLOCK_Y * 4):
        off0 = start + reduction_y
        off1 = off0 + BLOCK_Y
        off2 = off1 + BLOCK_Y
        off3 = off2 + BLOCK_Y
        acc0 = acc0 + tl.load(
            input_ptr + base + off0[:, None] * K + pid_k + x[None, :],
            mask=(off0 < N)[:, None],
            other=0.0,
        ).to(tl.float32)
        acc1 = acc1 + tl.load(
            input_ptr + base + off1[:, None] * K + pid_k + x[None, :],
            mask=(off1 < N)[:, None],
            other=0.0,
        ).to(tl.float32)
        acc2 = acc2 + tl.load(
            input_ptr + base + off2[:, None] * K + pid_k + x[None, :],
            mask=(off2 < N)[:, None],
            other=0.0,
        ).to(tl.float32)
        acc3 = acc3 + tl.load(
            input_ptr + base + off3[:, None] * K + pid_k + x[None, :],
            mask=(off3 < N)[:, None],
            other=0.0,
        ).to(tl.float32)
    lane_sum = acc0 + acc1
    lane_sum = lane_sum + acc2
    lane_sum = lane_sum + acc3
    total = tl.sum(lane_sum, axis=0)
    tl.store(output_ptr + pid_m * K + pid_k + x, total * (1.0 / N))


@libentry()
@triton.jit
def mean_dim_kernel_h100_non_inner_small_fp32(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
):
    """FP32 non-inner reduction when CUDA assigns one output per thread."""
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    base = pid_m * N * K + pid_k
    # This is Reduce.cuh's vt0=4 thread reduction: four independent
    # accumulators are combined after the full four-value chunks and tail.
    acc0 = tl.zeros((), dtype=tl.float32)
    acc1 = tl.zeros((), dtype=tl.float32)
    acc2 = tl.zeros((), dtype=tl.float32)
    acc3 = tl.zeros((), dtype=tl.float32)
    for start in range(0, (N // 4) * 4, 4):
        acc0 = acc0 + tl.load(input_ptr + base + start * K).to(tl.float32)
        acc1 = acc1 + tl.load(input_ptr + base + (start + 1) * K).to(tl.float32)
        acc2 = acc2 + tl.load(input_ptr + base + (start + 2) * K).to(tl.float32)
        acc3 = acc3 + tl.load(input_ptr + base + (start + 3) * K).to(tl.float32)
    tail_start = (N // 4) * 4
    if N % 4 > 0:
        acc0 = acc0 + tl.load(input_ptr + base + tail_start * K).to(tl.float32)
    if N % 4 > 1:
        acc1 = acc1 + tl.load(input_ptr + base + (tail_start + 1) * K).to(tl.float32)
    if N % 4 > 2:
        acc2 = acc2 + tl.load(input_ptr + base + (tail_start + 2) * K).to(tl.float32)
    total = acc0 + acc1
    total = total + acc2
    total = total + acc3
    tl.store(output_ptr + pid_m * K + pid_k, total * (1.0 / N))


@libentry()
@triton.jit
def mean_dim_kernel_h100_non_inner_global_fp32(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    CTAS: tl.constexpr,
):
    """Exact H100 global-split tree for a non-inner FP32 reduction.

    CUDA's ``Reduce.cuh`` gives every block-y lane four serial accumulators,
    stores one partial per CTA, then has the last CTA combine staging entries
    in increasing CTA order before the decreasing-offset block-y tree.  A
    single PPU program models those same scalar dependencies explicitly.  It
    is intentionally a correctness path for the small output shapes where
    CUDA enables global splitting (the Protenix ``[4,5370,9]`` case is the
    first one); it does not call any PPU-native reduction primitive.
    """
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    y = tl.arange(0, 16)
    base = pid_m * N * K + pid_k
    # For block=(32,16), CUDA's input index is
    #   cta * 16 + y + i * (16 * CTAS), i=0..3,
    # followed by another four-value group.  Keep one accumulator per
    # value-list slot so the combine order is identical.
    stride = 16 * CTAS
    global_sum = tl.zeros((16,), dtype=tl.float32)
    for cta in range(CTAS):
        idx0 = cta * 16 + y
        acc0 = tl.zeros((16,), dtype=tl.float32)
        acc1 = tl.zeros((16,), dtype=tl.float32)
        acc2 = tl.zeros((16,), dtype=tl.float32)
        acc3 = tl.zeros((16,), dtype=tl.float32)
        for start in range(0, N, 4 * 16 * CTAS):
            i0 = idx0 + start
            i1 = i0 + stride
            i2 = i1 + stride
            i3 = i2 + stride
            acc0 = acc0 + tl.load(
                input_ptr + base + i0 * K, mask=i0 < N, other=0.0
            ).to(tl.float32)
            acc1 = acc1 + tl.load(
                input_ptr + base + i1 * K, mask=i1 < N, other=0.0
            ).to(tl.float32)
            acc2 = acc2 + tl.load(
                input_ptr + base + i2 * K, mask=i2 < N, other=0.0
            ).to(tl.float32)
            acc3 = acc3 + tl.load(
                input_ptr + base + i3 * K, mask=i3 < N, other=0.0
            ).to(tl.float32)
        z = acc0 + acc1
        z = z + acc2
        z = z + acc3
        # H100 block_y_reduce uses offsets 8,4,2,1.  On PPU the identity
        # lane order is the matching tree (as established by the 16-lane
        # direct reduction probe), so tl.sum is safe here.
        local = tl.sum(z, axis=0)
        # The last CTA's thread y combines staging CTAs y, y+16, ... in
        # increasing order.  Only one vector slot is updated for each CTA;
        # this keeps the staging order explicit without tensor indexing.
        global_sum = global_sum + tl.where(
            y == (cta % 16), local, tl.zeros((16,), dtype=tl.float32)
        )
    # Final H100 block-y tree has the same 16-lane order.
    total = tl.sum(global_sum, axis=0)
    tl.store(output_ptr + pid_m * K + pid_k, total * (1.0 / N))


def _h100_next_pow2(value):
    result = 1
    while result < value:
        result <<= 1
    return result


def _h100_floor_pow2(value):
    result = 1
    while (result << 1) <= value:
        result <<= 1
    return result


def _h100_inner_block_width(M, N):
    """Return ReduceConfig's block-x width for a contiguous FP32 mean."""
    # CUDA's fastest-dimension vectorization packs four FP32 values when the
    # reduction is larger than one warp.  The config is built on packed
    # elements, then may widen the block after choosing block-y.
    dim0 = N // 4 if N > 128 else N
    dim0_pow2 = _h100_floor_pow2(max(dim0, 1))
    dim1_pow2 = _h100_floor_pow2(max(M, 1))
    initial_width = min(dim0_pow2, 32)
    block_height = min(dim1_pow2, 512 // initial_width)
    return min(dim0_pow2, 512 // block_height)


def _h100_non_inner_global_ctas(M, N, K, block_width, block_height):
    """Mirror CUDA's CTA split count for the non-inner FP32 path.

    The PPU correctness kernel currently models the CUDA configuration with
    one output scalar per lane (the Protenix K=3/9 layouts).  H100 exposes 132
    SMs and 2048 resident threads/SM; those are only used to reproduce the
    reference reduction order, never as a PPU performance assumption.
    """
    if block_width != 32 or block_height != 16:
        return 1
    grid = (M * K + block_width - 1) // block_width
    target_grid = 132 * (2048 // (block_width * block_height))
    values_per_thread = (N + block_height - 1) // block_height
    ctas1 = (target_grid + grid - 1) // grid
    ctas2 = (values_per_thread + 16 - 1) // 16
    ctas3 = (values_per_thread + 256 - 1) // 256
    return max(min(ctas1, ctas2), ctas3)


def _h100_mean_inner_candidate(inp, out, M, N):
    if (
        inp.dtype != torch.float32
        or not inp.is_contiguous()
        or N < 1
    ):
        return False
    block_x = _h100_inner_block_width(M, N)
    with torch_device_fn.device(inp.device):
        if block_x <= 32:
            if N > 128:
                mean_dim_kernel_h100_inner_fp32[(M,)](
                    out, inp, M, N=N, num_warps=1
                )
            else:
                mean_dim_kernel_h100_inner_small_fp32[(M,)](
                    out, inp, M, N=N, BLOCK_X=block_x, num_warps=1
                )
        elif block_x in (256, 512) and N > 128:
            # Preserve H100's per-warp accumulation and shared offset tree.
            # The 64/128-lane kernel cannot cover a 256-lane block.
            warps = block_x // 32
            partial = torch.empty((M, warps, 32), dtype=torch.float32, device=inp.device)
            mean_dim_kernel_h100_inner_wide_partial_fp32[(M, warps)](
                partial, inp, M, N=N, BLOCK_X=block_x,
                INPUT_SHIFT=(inp.data_ptr() // inp.element_size()) % 4, num_warps=1
            )
            mean_dim_kernel_h100_inner_wide_combine_fp32[(M,)](
                out, partial, M, N=N, BLOCK_X=block_x, num_warps=1
            )
        elif N > 128:
            mean_dim_kernel_h100_inner_wide_vec_fp32[(M,)](
                out, inp, M, N=N, BLOCK_X=block_x, num_warps=1
            )
        else:
            mean_dim_kernel_h100_inner_wide_scalar_fp32[(M,)](
                out, inp, M, N=N, BLOCK_X=block_x, num_warps=1
            )
    return True


def _h100_mean_non_inner_candidate(inp, out, M, N, K):
    if (
        inp.dtype != torch.float32
        or not inp.is_contiguous()
        or N < 1
        or K < 1
        or K > 32
    ):
        return False
    # CUDA's ReduceConfig sizes the two block dimensions from the flattened
    # output count (M*K) and the reduction length.  For this scalar-output
    # kernel we keep the same block-y choice; e.g. M*K=3 -> 256 lanes and
    # M*K=12 -> 64 lanes with the CUDA reduction's 512-thread cap.
    block_width = min(_h100_floor_pow2(M * K), 32)
    block_height = min(_h100_floor_pow2(N), 512 // block_width)
    if N < min(block_height * 16, 256):
        with torch_device_fn.device(inp.device):
            mean_dim_kernel_h100_non_inner_small_fp32[(M, K)](
                out, inp, M, N=N, K=K, num_warps=1
            )
        return True
    # H100 enables a global staging reduction for the common K=3/9, block
    # (32,16) layouts.  Reproduce its per-CTA and final staging trees in pure
    # Triton; do not route through the PPU-native mean implementation.
    if block_height == 16 and K in (3, 9):
        ctas = _h100_non_inner_global_ctas(M, N, K, block_width, block_height)
        if ctas > 1:
            with torch_device_fn.device(inp.device):
                mean_dim_kernel_h100_non_inner_global_fp32[(M, K)](
                    out,
                    inp,
                    M,
                    N=N,
                    K=K,
                    CTAS=ctas,
                    num_warps=1,
                )
            return True
    if block_height < 32 or N < min(block_height * 16, 256):
        return False
    with torch_device_fn.device(inp.device):
        mean_dim_kernel_h100_non_inner_fp32[(M, K)](
            out,
            inp,
            M,
            N=N,
            K=K,
            BLOCK_Y=block_height,
            num_warps=block_height // 32,
        )
    return True


@libentry()
@triton.jit
def mean_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    # accumulation dtype
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    if tl.constexpr(mid.dtype.element_ty == tl.float16) or tl.constexpr(
        mid.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = mid.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(mid_val)
    # divide by total element count M to get mean
    mean_val = sum_val / M
    tl.store(out, mean_val)


def mean(inp, *, dtype=None):
    logger.debug("GEMS MEAN")
    source_inp = inp
    M = source_inp.numel()
    if dtype is None:
        dtype = source_inp.dtype
    if M == 0:
        if not (dtype.is_floating_point or dtype.is_complex):
            raise RuntimeError("mean(): could not infer output dtype")
        return torch.full([], float("nan"), dtype=dtype, device=source_inp.device)
    inp = source_inp.contiguous()
    if dtype == torch.float32 and source_inp.is_contiguous():
        out = torch.empty([], dtype=dtype, device=inp.device)
        if _h100_mean_inner_candidate(inp.reshape(1, M), out, 1, M):
            return out
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid)
    return out


@libentry()
@triton.jit
def mean_dim_kernel_non_inner_vec(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_K: tl.constexpr,  # number of threads per block along K
    VEC_SIZE: tl.constexpr,  # elements per thread (1 for FP32, 8 for FP16/BF16)
):
    # Determine accumulation and load behavior
    input_dtype = input_ptr.dtype.element_ty
    if tl.constexpr(input_dtype == tl.float16) or tl.constexpr(
        input_dtype == tl.bfloat16
    ):
        ACC_DTYPE = tl.float32
        # VEC_SIZE should be 4 or 8 for vectorization
    else:
        ACC_DTYPE = input_dtype
        # VEC_SIZE = 1 for FP32

    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)

    # Each thread handles VEC_SIZE consecutive elements
    k_base = pid_k * BLOCK_SIZE_K * VEC_SIZE
    k_offsets = (
        k_base
        + tl.arange(0, BLOCK_SIZE_K)[:, None] * VEC_SIZE
        + tl.arange(0, VEC_SIZE)[None, :]
    )
    # Shape: [BLOCK_SIZE_K, VEC_SIZE]
    k_mask = k_offsets < K

    # Accumulator: [BLOCK_SIZE_K, VEC_SIZE]
    acc = tl.zeros((BLOCK_SIZE_K, VEC_SIZE), dtype=ACC_DTYPE)

    base = pid_m * N * K

    for n in range(N):
        offsets = base + n * K + k_offsets
        # This will trigger vectorized load if VEC_SIZE >= 4 and aligned
        val = tl.load(input_ptr + offsets, mask=k_mask, other=0.0)
        acc += val.to(ACC_DTYPE)

    mean_val = acc / N

    # Store back
    out_offsets = pid_m * K + k_offsets
    tl.store(output_ptr + out_offsets, mean_val, mask=k_mask)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("mean_non_inner"))
@triton.jit
def mean_dim_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # accumulation dtype
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        input_ptrs = input_ptr + inp_offset
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        # sum along reduction axis (N) -> keep dims so axis 0 corresponds to TILE_K
        summed = tl.sum(inp, axis=0, keep_dims=True)
        # divide by N to get mean
        out = summed / N
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)
    else:
        sum_tile = tl.zeros([TILE_N, TILE_K], dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            sum_tile += inp
        summed = tl.sum(sum_tile, axis=0, keep_dims=True)
        out = summed / N
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def mean_dim_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = tle.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        inp_offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + inp_offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        summed = tl.sum(inp, axis=0)
        out = summed / N
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)
    else:
        sum_vec = tl.zeros(
            [
                TILE_N,
            ],
            dtype=cdtype,
        )
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_offsets = pid_m * N + n_offsets
            mask = n_offsets < N
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            sum_vec += inp
        summed = tl.sum(sum_vec, axis=0)
        out = summed / N
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def mean_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    # Map the program id to the row of inp it should compute.
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0).to(cdtype)
        _sum += a
    summed = tl.sum(_sum, axis=1)[:, None]
    mean = summed / N
    tl.store(out, mean, row_mask)


def mean_dim_comm(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    logger.debug("GEMS MEAN_DIM")
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64

    # Preserve the original view for the contiguous-layout eligibility check;
    # the actual reduction is always implemented by FlagGems kernels here.
    source_inp = inp

    if dim == []:
        # mean over all elements
        if not keepdim:
            return mean(inp, dtype=dtype)
        else:
            dim_num = inp.ndim
            return torch.reshape(mean(inp, dtype=dtype), [1] * dim_num)

    shape = list(inp.shape)

    # -------- normalize dim to a list of ints --------
    if isinstance(dim, int):
        dim = [dim]
    else:
        try:
            dim = list(dim)
        except TypeError:
            raise TypeError(
                f"dim must be an int, iterable of ints, or [], got {type(dim)}"
            )

    dim = [d % inp.ndim for d in dim]
    # -------------------------------------------------

    if len(dim) == 1:
        dim0 = dim[0]
        N = inp.shape[dim0]  # reduction length
        # product of dims before dim0; use initializer 1 for empty slice
        M = reduce(lambda x, y: x * y, shape[:dim0], 1)
        inp = inp.contiguous()
        shape[dim0] = 1
        if N == 0:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise RuntimeError("mean(): could not infer output dtype")
            out = torch.full(shape, float("nan"), dtype=dtype, device=inp.device)
            if not keepdim:
                out = out.squeeze(dim=dim0)
            return out
        out_was_provided = out is not None
        # There is no row to reduce when a non-reduced prefix dimension is
        # empty.  Handle it before the old K = numel // M // N calculation.
        if M == 0:
            if out is None:
                out = torch.empty(shape, dtype=dtype, device=inp.device)
            if not keepdim:
                out = out.squeeze(dim=dim0)
            return out
        K = inp.numel() // M // N
        if not out_was_provided:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        # The H100 reference uses a different reduction tree from PPU's
        # generic ``tl.sum``.  Handle the large FP32 non-inner reductions used
        # by Protenix with the explicit H100-order kernel.
        if (
            not out_was_provided
            and source_inp.is_contiguous()
            and dtype == torch.float32
            and dim0 == inp.ndim - 1
            and _h100_mean_inner_candidate(inp.reshape(M, N), out.reshape(M), M, N)
        ):
            if not keepdim:
                out = out.squeeze(dim=dim0)
            return out

        if (
            not out_was_provided
            and source_inp.is_contiguous()
            and dtype == torch.float32
            and dim0 != inp.ndim - 1
            and _h100_mean_non_inner_candidate(inp.reshape(M, N, K), out, M, N, K)
        ):
            if not keepdim:
                out = out.squeeze(dim=dim0)
            return out

        with torch_device_fn.device(inp.device):
            if K >= 1024:
                input_dtype = inp.dtype
                if input_dtype in (torch.float16, torch.bfloat16):
                    VEC_SIZE = 8
                    BLOCK_SIZE_K = 128
                else:
                    VEC_SIZE = 1
                    BLOCK_SIZE_K = min(triton.next_power_of_2(K), 512)
                grid = (M, triton.cdiv(K, BLOCK_SIZE_K * VEC_SIZE))
                mean_dim_kernel_non_inner_vec[grid](
                    out,
                    inp,
                    M,
                    N,
                    K,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    VEC_SIZE=VEC_SIZE,
                    num_warps=8 if BLOCK_SIZE_K <= 128 else 16,
                )
            elif K > 1:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                mean_dim_kernel_non_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                    K,
                )
            else:
                grid = (M, 1, 1)
                mean_dim_kernel_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                )
        if not keepdim:
            out = out.squeeze(dim=dim0)
        return out
    else:
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        if N == 0:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise RuntimeError("mean(): could not infer output dtype")
            out = torch.full(shape, float("nan"), dtype=dtype, device=inp.device)
            if not keepdim:
                out = out.squeeze(dim=dim)
            return out
        inp = dim_compress(inp, dim)
        M = inp.numel() // N
        if out is None:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            mean_dim_kernel[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out


def mean_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS MEAN_DIM (wrapper)")

    return mean_dim_comm(inp, dim, keepdim, dtype=dtype)
