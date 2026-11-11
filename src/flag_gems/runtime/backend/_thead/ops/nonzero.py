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

"""Block-local compaction for the Tensor-returning PPU nonzero entry."""

import torch
import triton
import triton.language as tl

from flag_gems.ops.nonzero import nonzero as _default_nonzero
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import full_write_allocation, libentry

_MAX_NONZERO_BLOCK = 8192
_REAL_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
                torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64)


@libentry()
@triton.jit
def _small_nonzero(inp, out, count, n, s0, s1, s2, s3, NDIM: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    valid = lane < n
    x = tl.load(inp + lane, valid, other=0)
    if x.dtype.is_floating():
        if x.dtype.primitive_bitwidth == 64:
            nonzero_value = (x.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF) != 0
        elif x.dtype.primitive_bitwidth == 32:
            nonzero_value = (x.to(tl.int32, bitcast=True) & 0x7FFFFFFF) != 0
        else:
            nonzero_value = (x.to(tl.int16, bitcast=True) & 0x7FFF) != 0
    else:
        nonzero_value = x != 0
    selected = valid & nonzero_value
    position = tl.cumsum(selected.to(tl.int32)) - 1
    total = tl.sum(selected.to(tl.int32))
    # Initialize the complete overallocated output in this same kernel. The
    # barrier separates initialization from stable compaction across warps.
    for init_axis in tl.static_range(NDIM):
        tl.store(out + lane * NDIM + init_axis, 0, valid)
    tl.debug_barrier()
    flat = lane
    if NDIM >= 4:
        coordinate = flat % s3
        flat = flat // s3
        tl.store(out + position * NDIM + 3, coordinate.to(tl.int64), selected)
    if NDIM >= 3:
        coordinate = flat % s2
        flat = flat // s2
        tl.store(out + position * NDIM + 2, coordinate.to(tl.int64), selected)
    if NDIM >= 2:
        coordinate = flat % s1
        flat = flat // s1
        tl.store(out + position * NDIM + 1, coordinate.to(tl.int64), selected)
    tl.store(out + position * NDIM, flat.to(tl.int64), selected)
    tl.store(count, total)


def nonzero(inp, *, as_tuple=False):
    if (not inp.is_cuda or inp.dtype not in _REAL_DTYPES or not inp.is_contiguous()
            or not 1 <= inp.ndim <= 4 or inp.numel() > _MAX_NONZERO_BLOCK):
        return _default_nonzero(inp, as_tuple=as_tuple)
    n, ndim = inp.numel(), inp.ndim
    output = full_write_allocation.empty((n, ndim), dtype=torch.int64, device=inp.device)
    if n:
        count = full_write_allocation.empty((1,), dtype=torch.int32, device=inp.device)
        block = triton.next_power_of_2(max(n, 1))
        shape = list(inp.shape) + [1] * (4 - ndim)
        with torch_device_fn.device(inp.device):
            _small_nonzero[(1,)](inp, output, count, n, *shape, NDIM=ndim,
                                BLOCK=block, num_warps=4 if block <= 4096 else 8)
        nnz = int(count.item())
    else:
        nnz = 0
    result = output.as_strided((nnz, ndim), (ndim, 1))
    return tuple(result.select(1, axis) for axis in range(ndim)) if as_tuple else result
