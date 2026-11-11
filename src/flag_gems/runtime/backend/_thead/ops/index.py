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

"""Compact a short boolean mask and gather its values in one PPU launch."""

import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.index import index as _default_index
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import full_write_allocation, libentry

_COPY_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
                torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64)
_MAX_MASK_BLOCK = 4096
_MAX_OUTPUT_CAPACITY = 1 << 20
_BLOCK_ELEMENTS = 4096


@libentry()
@triton.jit
def _boolean_index(
    inp, mask, out, count, nmask, other, inner, capacity,
    n0, n1, n2, n3, s0, s1, s2, s3,
    NDIM: tl.constexpr, BLOCK_MASK: tl.constexpr, BLOCK_OTHER: tl.constexpr,
):
    row = tl.arange(0, BLOCK_MASK)
    selected = tl.load(mask + row, row < nmask, other=0)
    position = tl.cumsum(selected.to(tl.int32)) - 1
    nnz = tl.sum(selected.to(tl.int32))
    column = tl.program_id(0) * BLOCK_OTHER + tl.arange(0, BLOCK_OTHER)
    outer = column // inner
    suffix = column % inner
    source_flat = (outer[None, :] * nmask + row[:, None]) * inner + suffix[None, :]
    flat = source_flat
    offset = tl.full((BLOCK_MASK, BLOCK_OTHER), 0, tl.int64)
    if NDIM >= 4:
        offset += (flat % n3).to(tl.int64) * s3
        flat = flat // n3
    if NDIM >= 3:
        offset += (flat % n2).to(tl.int64) * s2
        flat = flat // n2
    if NDIM >= 2:
        offset += (flat % n1).to(tl.int64) * s1
        flat = flat // n1
    offset += flat.to(tl.int64) * s0
    target = (outer[None, :] * nnz + position[:, None]) * inner + suffix[None, :]
    valid = selected[:, None] & (column[None, :] < other)
    # Integer loads/stores copy every bit, including signed zero, NaN payloads
    # and subnormals. No arithmetic is applied to the gathered values.
    if inp.dtype.element_ty.primitive_bitwidth == 64:
        source_ptr = inp.to(tl.pointer_type(tl.int64))
        target_ptr = out.to(tl.pointer_type(tl.int64))
    elif inp.dtype.element_ty.primitive_bitwidth == 32:
        source_ptr = inp.to(tl.pointer_type(tl.int32))
        target_ptr = out.to(tl.pointer_type(tl.int32))
    elif inp.dtype.element_ty.primitive_bitwidth == 16:
        source_ptr = inp.to(tl.pointer_type(tl.int16))
        target_ptr = out.to(tl.pointer_type(tl.int16))
    else:
        source_ptr = inp.to(tl.pointer_type(tl.int8))
        target_ptr = out.to(tl.pointer_type(tl.int8))
    values = tl.load(source_ptr + offset, valid, other=0)
    tl.store(target_ptr + target, values, valid)
    # Selected values cover the compact prefix. Zero only the disjoint unused
    # tail so the entire overallocated buffer is written, without a barrier.
    unused = row[:, None] * other + column[None, :]
    tl.store(target_ptr + unused, 0,
             (row[:, None] < nmask) & (column[None, :] < other)
             & (unused >= nnz * other) & (unused < capacity))
    if tl.program_id(0) == 0:
        tl.store(count, nnz)


def index(inp, indices):
    if (not inp.is_cuda or inp.layout != torch.strided or inp.dtype not in _COPY_DTYPES
            or not 1 <= inp.ndim <= 4 or not 0 < inp.numel() <= _MAX_OUTPUT_CAPACITY):
        return _default_index(inp, indices)
    masks = [(axis, value) for axis, value in enumerate(indices) if value is not None]
    if len(masks) != 1:
        return _default_index(inp, indices)
    axis, mask = masks[0]
    if (mask.dtype != torch.bool or mask.device != inp.device or not mask.is_contiguous()
            or not 1 <= mask.ndim <= inp.ndim - axis
            or len(indices) - 1 + mask.ndim > inp.ndim
            or tuple(mask.shape) != tuple(inp.shape[axis:axis + mask.ndim])):
        return _default_index(inp, indices)
    nmask = mask.numel()
    if not 0 < nmask <= _MAX_MASK_BLOCK:
        return _default_index(inp, indices)
    other = inp.numel() // nmask
    inner = math.prod(inp.shape[axis + mask.ndim:])
    block_mask = triton.next_power_of_2(nmask)
    block_other = min(16, _BLOCK_ELEMENTS // block_mask)
    output = full_write_allocation.empty((inp.numel(),), dtype=inp.dtype, device=inp.device)
    count = full_write_allocation.empty((1,), dtype=torch.int32, device=inp.device)
    shape = list(inp.shape) + [1] * (4 - inp.ndim)
    strides = list(inp.stride()) + [1] * (4 - inp.ndim)
    with torch_device_fn.device(inp.device):
        _boolean_index[(triton.cdiv(other, block_other),)](
            inp, mask, output, count, nmask, other, inner, inp.numel(),
            *shape, *strides, NDIM=inp.ndim,
            BLOCK_MASK=block_mask, BLOCK_OTHER=block_other, num_warps=4,
        )
    nnz = int(count.item())
    shape = list(inp.shape[:axis]) + [nnz] + list(inp.shape[axis + mask.ndim:])
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return output.as_strided(shape, strides)
