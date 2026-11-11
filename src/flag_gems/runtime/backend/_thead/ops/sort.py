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

"""Stable short-row integer sorting with one block per row on PPU."""

import torch
import triton
import triton.language as tl

from flag_gems.ops.sort import sort as _default_sort
from flag_gems.ops.sort import sort_stable as _default_sort_stable
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import full_write_allocation, libentry

_INTEGER_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
# The network holds one value and one int32 source index per lane. Bound the
# single-block register/exchange working set; longer rows keep radix sorting.
_MAX_SORT_BLOCK = 4096


@triton.jit
def _partner(values, AXIS: tl.constexpr, LOG_BLOCK: tl.constexpr):
    shape: tl.constexpr = [2] * LOG_BLOCK
    cube = tl.reshape(values, shape)
    # The compiler's bitonic sort uses this XOR reduction to exchange a pair
    # along a two-element axis. No floating conversion or dynamic gather.
    other = cube ^ tl.xor_sum(cube, axis=AXIS, keep_dims=True)
    return tl.reshape(other, values.shape)


@libentry()
@triton.jit
def _short_integer_sort(
    inp, out, out_indices, n, input_step, output_step,
    ROW_SHAPE: tl.constexpr, ROW_INPUT_STRIDES: tl.constexpr,
    ROW_OUTPUT_STRIDES: tl.constexpr, BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr, DESCENDING: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    input_base = tl.full((), 0, tl.int64)
    output_base = tl.full((), 0, tl.int64)
    for axis in tl.static_range(len(ROW_SHAPE) - 1, -1, -1):
        coordinate = row % ROW_SHAPE[axis]
        row = row // ROW_SHAPE[axis]
        input_base += coordinate * ROW_INPUT_STRIDES[axis]
        output_base += coordinate * ROW_OUTPUT_STRIDES[axis]

    lane = tl.arange(0, BLOCK)
    ids = lane
    values = tl.load(inp + input_base + lane * input_step, lane < n, other=0)
    if inp.dtype.element_ty == tl.int64:
        values = values.to(tl.int64)
    else:
        values = values.to(tl.int32)
    # Compare (invalid, value in requested order, original position). All
    # comparisons stay integer, including int64 extremes and equal values.
    for stage in tl.static_range(1, LOG_BLOCK + 1):
        for step in tl.static_range(stage):
            distance = 1 << (stage - 1 - step)
            other = _partner(values, AXIS=LOG_BLOCK - stage + step, LOG_BLOCK=LOG_BLOCK)
            other_ids = _partner(ids, AXIS=LOG_BLOCK - stage + step, LOG_BLOCK=LOG_BLOCK)
            valid, other_valid = ids < n, other_ids < n
            if DESCENDING:
                greater_value = values < other
            else:
                greater_value = values > other
            greater = ((~valid) & other_valid) | (
                (valid == other_valid) & (greater_value | ((values == other) & (ids > other_ids)))
            )
            swap = greater ^ ((lane & distance) != 0) ^ ((lane & (1 << stage)) != 0)
            values = tl.where(swap, other, values)
            ids = tl.where(swap, other_ids, ids)
    tl.store(out + output_base + lane * output_step, values, lane < n)
    tl.store(out_indices + output_base + lane * output_step, ids.to(tl.int64), lane < n)


def _try_short_sort(inp, dim, descending):
    if inp.dtype not in _INTEGER_DTYPES or inp.layout != torch.strided or not inp.is_cuda:
        return None
    rank = max(inp.ndim, 1)
    if not -rank <= dim < rank or rank > 8:
        return None
    dim %= rank
    shape = tuple(inp.shape) if inp.ndim else (1,)
    strides = tuple(inp.stride()) if inp.ndim else (1,)
    n = shape[dim]
    block = triton.next_power_of_2(max(n, 1))
    if block > _MAX_SORT_BLOCK:
        return None
    output = full_write_allocation.empty(tuple(inp.shape), dtype=inp.dtype, device=inp.device)
    indices = full_write_allocation.empty(tuple(inp.shape), dtype=torch.int64, device=inp.device)
    if not inp.numel():
        return output, indices
    output_strides = [1] * rank
    for axis in range(rank - 2, -1, -1):
        output_strides[axis] = output_strides[axis + 1] * shape[axis + 1]
    row_shape = tuple(shape[a] for a in range(rank) if a != dim)
    row_input_strides = tuple(strides[a] for a in range(rank) if a != dim)
    row_output_strides = tuple(output_strides[a] for a in range(rank) if a != dim)
    with torch_device_fn.device(inp.device):
        _short_integer_sort[(inp.numel() // n,)](
            inp, output, indices, n, strides[dim], output_strides[dim],
            ROW_SHAPE=row_shape, ROW_INPUT_STRIDES=row_input_strides,
            ROW_OUTPUT_STRIDES=row_output_strides, BLOCK=block,
            LOG_BLOCK=block.bit_length() - 1, DESCENDING=descending,
            num_warps=4 if block <= 1024 else 8,
        )
    return output, indices


def sort(inp, dim=-1, descending=False):
    result = _try_short_sort(inp, dim, descending)
    return _default_sort(inp, dim=dim, descending=descending) if result is None else result


def sort_stable(inp, *, stable, dim=-1, descending=False):
    result = _try_short_sort(inp, dim, descending)
    return _default_sort_stable(inp, stable=stable, dim=dim, descending=descending) if result is None else result
