"""Copy dense FP32 permutations using coalesced loads and a layout exchange."""
import os

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from flag_gems.ops.tf32_precision import empty_full_write
from flag_gems.runtime import torch_device_fn

from .bmm_rne_layout import _layout_spec

ENABLED = os.environ.get("FLAG_GEMS_PPU_BMM_DENSE_COPY", "1") != "0"


@gluon.jit
def _copy_transpose(
    X, Y, M: gl.constexpr, N: gl.constexpr,
    S0: gl.constexpr, S1: gl.constexpr, S2: gl.constexpr,
    O0: gl.constexpr, O1: gl.constexpr, O2: gl.constexpr,
    TILE: gl.constexpr,
):
    source_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [1, 4], [0, 1])
    output_layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    rows = gl.program_id(0) * TILE + gl.arange(0, TILE, layout=gl.SliceLayout(1, source_layout))
    cols = gl.program_id(1) * TILE + gl.arange(0, TILE, layout=gl.SliceLayout(0, source_layout))
    batch = gl.program_id(2)
    value = gl.load(
        X + batch * S2 + rows[:, None] * S0 + cols[None, :] * S1,
        (rows[:, None] < M) & (cols[None, :] < N), other=0.0,
    )
    value = gl.convert_layout(value, output_layout)
    out_rows = gl.program_id(0) * TILE + gl.arange(0, TILE, layout=gl.SliceLayout(1, output_layout))
    out_cols = gl.program_id(1) * TILE + gl.arange(0, TILE, layout=gl.SliceLayout(0, output_layout))
    gl.store(
        Y + batch * O2 + out_rows[:, None] * O0 + out_cols[None, :] * O1,
        value, (out_rows[:, None] < M) & (out_cols[None, :] < N),
    )


class _DenseCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, spec):
        _, rows, cols, batch, *strides = spec
        output = empty_full_write(value.shape, dtype=value.dtype, device=value.device)
        with torch_device_fn.device(value.device):
            _copy_transpose[(triton.cdiv(rows, 64), triton.cdiv(cols, 64), batch)](
                value, output, rows, cols, *strides, 64, num_warps=4,
            )
        return output

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def try_copy(value):
    if not ENABLED or not value.is_cuda or value.dtype != torch.float32 or value.is_neg():
        return None
    spec = _layout_spec(tuple(value.shape), tuple(value.stride()))
    if spec is None or spec[0] != "transpose":
        return None
    return _DenseCopy.apply(value, spec)
