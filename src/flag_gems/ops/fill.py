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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_scalar_func(inp, value_scalar):
    return tl.full(inp.shape, value_scalar, dtype=inp.dtype)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_tensor_func(inp, value):
    return value


@triton.jit
def fill_scalar_2d_strided_kernel(
    out,
    value,
    n0: tl.constexpr,
    n1: tl.constexpr,
    stride0: tl.constexpr,
    stride1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = offsets // n1
    cols = offsets % n1
    tl.store(out + rows * stride0 + cols * stride1, value, mask=offsets < n0 * n1)


@triton.jit
def fill_scalar_3d_strided_kernel(
    out,
    value,
    n0: tl.constexpr,
    n1: tl.constexpr,
    n2: tl.constexpr,
    stride0: tl.constexpr,
    stride1: tl.constexpr,
    stride2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    plane = n1 * n2
    index0 = offsets // plane
    remainder = offsets % plane
    index1 = remainder // n2
    index2 = remainder % n2
    pointers = out + index0 * stride0 + index1 * stride1 + index2 * stride2
    tl.store(pointers, value, mask=offsets < n0 * plane)


@triton.jit
def fill_scalar_4d_strided_kernel(
    out,
    value,
    n0: tl.constexpr,
    n1: tl.constexpr,
    n2: tl.constexpr,
    n3: tl.constexpr,
    stride0: tl.constexpr,
    stride1: tl.constexpr,
    stride2: tl.constexpr,
    stride3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    volume123 = n1 * n2 * n3
    index0 = offsets // volume123
    remainder = offsets % volume123
    plane23 = n2 * n3
    index1 = remainder // plane23
    remainder = remainder % plane23
    index2 = remainder // n3
    index3 = remainder % n3
    pointers = (
        out
        + index0 * stride0
        + index1 * stride1
        + index2 * stride2
        + index3 * stride3
    )
    tl.store(pointers, value, mask=offsets < n0 * volume123)


def fill_scalar(input, value):
    logger.debug("GEMS FILL (Dynamic)")
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_scalar_func(input, value, out0=out)


def fill_scalar_out(input, value, *, out=None):
    logger.debug("GEMS FILL_SCALAR_OUT")
    if out is None:
        return fill_scalar(input, value)
    with torch_device_fn.device(input.device):
        fill_scalar_func(input, value, out0=out)
    return out


def fill_tensor(input, value):
    if not value.is_cuda:
        return fill_scalar(input, value.item())
    logger.debug("GEMS FILL (Dynamic)")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_tensor_func(input, value, out0=out)


def fill_tensor_out(input, value, *, out=None):
    logger.debug("GEMS FILL_TENSOR_OUT")
    if out is None:
        return fill_tensor(input, value)
    if not value.is_cuda:
        return fill_scalar_out(input, value.item(), out=out)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    with torch_device_fn.device(input.device):
        fill_tensor_func(input, value, out0=out)
    return out


def fill_tensor_(self, value):
    if not value.is_cuda:
        return fill_scalar_(self, value.item())
    logger.debug("GEMS FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    with torch_device_fn.device(self.device):
        fill_tensor_func(self, value, out0=self)
    return self


def fill_scalar_(self, value):
    logger.debug("GEMS FILL_SCALAR_")
    with torch_device_fn.device(self.device):
        # pointwise_dynamic currently returns one tile size for some rank-2
        # strided views while its generated wrapper expects two.  Use an
        # explicit stride-aware FlagGems kernel for that common in-place case.
        if self.ndim in (2, 3, 4):
            if self.numel() == 0:
                return self
            grid = (triton.cdiv(self.numel(), 256),)
            if self.ndim == 2:
                fill_scalar_2d_strided_kernel[grid](
                    self,
                    value,
                    self.shape[0],
                    self.shape[1],
                    self.stride(0),
                    self.stride(1),
                    BLOCK_SIZE=256,
                )
            elif self.ndim == 3:
                fill_scalar_3d_strided_kernel[grid](
                    self,
                    value,
                    *self.shape,
                    *self.stride(),
                    BLOCK_SIZE=256,
                )
            else:
                fill_scalar_4d_strided_kernel[grid](
                    self,
                    value,
                    *self.shape,
                    *self.stride(),
                    BLOCK_SIZE=256,
                )
            return self
        fill_scalar_func(self, value, out0=self)
    return self
