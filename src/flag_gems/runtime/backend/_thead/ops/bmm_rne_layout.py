# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0

"""Prepare RNE TF32 operands without strided scalar gathers.

The caller keeps its existing dot tiles, K traversal and accumulator. Each
operand is rounded once into contiguous storage. Dense permutations load in
physical order and exchange a tile through shared memory before storing in
logical order.
"""

import math
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from flag_gems.ops.tf32_precision import empty_full_write, round_tf32_rne
from flag_gems.runtime import torch_device_fn


@triton.jit
def _round_flat(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0.0)
    tl.store(Y + offsets, round_tf32_rne(x, True), offsets < N)


@gluon.jit
def _round_transpose(
    X, Y, M: gl.constexpr, N: gl.constexpr,
    S0: gl.constexpr, S1: gl.constexpr, S2: gl.constexpr,
    O0: gl.constexpr, O1: gl.constexpr, O2: gl.constexpr,
    TILE: gl.constexpr,
):
    source_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [1, 4], [0, 1])
    output_layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    rows = gl.program_id(0) * TILE + gl.arange(
        0, TILE, layout=gl.SliceLayout(1, source_layout)
    )
    cols = gl.program_id(1) * TILE + gl.arange(
        0, TILE, layout=gl.SliceLayout(0, source_layout)
    )
    batch = gl.program_id(2)
    x = gl.load(
        X + batch * S2 + rows[:, None] * S0 + cols[None, :] * S1,
        (rows[:, None] < M) & (cols[None, :] < N),
        other=0.0,
    )
    bits = x.to(gl.uint32, bitcast=True)
    finite = (bits & 0x7F800000) != 0x7F800000
    rounded = gl.inline_asm_elementwise(
        "cvt.rn.tf32.f32 $0, $1;", "=f,f", [x], gl.float32, True, 1
    )
    # Match round_tf32_rne, including infinity, NaN sign and payload.
    rounded = gl.where(finite, rounded.to(gl.uint32, bitcast=True), bits).to(
        gl.float32, bitcast=True
    )
    rounded = gl.convert_layout(rounded, output_layout)
    out_rows = gl.program_id(0) * TILE + gl.arange(
        0, TILE, layout=gl.SliceLayout(1, output_layout)
    )
    out_cols = gl.program_id(1) * TILE + gl.arange(
        0, TILE, layout=gl.SliceLayout(0, output_layout)
    )
    gl.store(
        Y + batch * O2 + out_rows[:, None] * O0 + out_cols[None, :] * O1,
        rounded,
        (out_rows[:, None] < M) & (out_cols[None, :] < N),
    )


@lru_cache(maxsize=256)
def _layout_spec(shape, strides):
    if len(shape) != 3 or not 0 < math.prod(shape) < 2**31:
        return None
    expected = 1
    for axis in sorted(range(3), key=lambda i: strides[i]):
        if shape[axis] == 1:
            continue
        if strides[axis] != expected:
            return None
        expected *= shape[axis]
    inner = min(range(3), key=lambda i: strides[i])
    contiguous = (shape[1] * shape[2], shape[2], 1)
    if inner == 2:
        if all(n == 1 or s == c for n, s, c in zip(shape, strides, contiguous)):
            return ("flat",)
        return None
    outer = 1 - inner
    return (
        "transpose", shape[inner], shape[2], shape[outer],
        strides[inner], strides[2], strides[outer],
        contiguous[inner], 1, contiguous[outer],
    )


def _round_dense(x):
    spec = _layout_spec(tuple(x.shape), tuple(x.stride()))
    assert spec is not None
    result = empty_full_write(x.shape, dtype=x.dtype, device=x.device)
    if spec[0] == "flat":
        _round_flat[(triton.cdiv(x.numel(), 8192),)](
            x, result, x.numel(), 8192, num_warps=8
        )
    else:
        _, rows, cols, batch, *strides = spec
        _round_transpose[(triton.cdiv(rows, 64), triton.cdiv(cols, 64), batch)](
            x, result, rows, cols, *strides, 64, num_warps=4
        )
    return result


def try_layout_bmm(a, b, out, dot_kernel):
    """Return False when the existing operand preparation should be retained."""
    if (
        not a.is_cuda or not b.is_cuda or a.device != b.device
        or a.dtype != torch.float32 or b.dtype != torch.float32
        or a.ndim != 3 or b.ndim != 3
        or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]
        or a.numel() + b.numel() < 2**24
        or a.is_neg() or b.is_neg()
    ):
        return False
    specs = [_layout_spec(tuple(x.shape), tuple(x.stride())) for x in (a, b)]
    if any(spec is None for spec in specs):
        return False
    if not any(spec[0] == "transpose" and min(spec[1], spec[2]) >= 32 for spec in specs):
        return False
    batch, rows, reduction = a.shape
    cols = b.shape[2]
    tile_m = 16 if rows < 16 else 32 if rows < 256 else 64
    tile_n = 16 if cols <= 16 else 32 if cols < 256 else 64
    with torch_device_fn.device(a.device):
        rounded_a, rounded_b = _round_dense(a), _round_dense(b)
        dot_kernel[(triton.cdiv(rows, tile_m), triton.cdiv(cols, tile_n), batch)](
            rounded_a, rounded_b, out, rows, cols, reduction, batch,
            *rounded_a.stride(), *rounded_b.stride(), *out.stride(),
            tile_m, tile_n, 32, num_warps=4,
        )
    return True
