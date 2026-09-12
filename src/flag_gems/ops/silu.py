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
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.triton_lang_extension import div_rn

logger = logging.getLogger(__name__)


@triton.jit
def _silu_contiguous_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    """Existing measured PPU path for the [*, 1024] training family."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    y = tl.fdiv(x_fp32, 1.0 + tl.exp(-x_fp32))
    tl.store(out_ptr + offsets, y.to(x.dtype), mask=mask)


@triton.jit
def _silu_backward_contiguous_kernel(
    x_ptr, dy_ptr, dx_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    """Existing measured PPU backward path for the [*, 1024] family."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sigma = tl.fdiv(1.0, 1.0 + tl.exp(-x))
    dx = dy * sigma * (1.0 + x * (1.0 - sigma))
    tl.store(dx_ptr + offsets, dx, mask=mask)


def _can_use_measured_ppu_path(x, env_name="FLAG_GEMS_PPU_SILU_BLOCK"):
    """Guard the previously measured 8192-element PPU schedule."""
    return (
        os.getenv(env_name) == "8192"
        and x.device.type == "cuda"
        and x.dtype == torch.float32
        and x.is_contiguous()
        and x.ndim >= 1
        and x.shape[-1] == 1024
        and x.numel() >= (1 << 20)
        # The measured kernel writes an output tensor directly and does not
        # install a backward node.  Restrict it to inference so an opt-in
        # benchmark flag cannot silently drop SiLU gradients in training.
        and not (torch.is_grad_enabled() and x.requires_grad)
    )


def _silu_measured_ppu(x, out=None):
    if out is None:
        out = torch.empty_like(x)
    n_elements = int(x.numel())
    grid = (triton.cdiv(n_elements, 8192),)
    with torch_device_fn.device(x.device):
        _silu_contiguous_kernel[grid](
            x, out, n_elements, BLOCK_SIZE=8192, num_warps=16
        )
    return out


def _silu_backward_measured_ppu(grad_output, self):
    grad_input = torch.empty_like(self)
    n_elements = int(self.numel())
    grid = (triton.cdiv(n_elements, 8192),)
    with torch_device_fn.device(self.device):
        _silu_backward_contiguous_kernel[grid](
            self,
            grad_output,
            grad_input,
            n_elements,
            BLOCK_SIZE=8192,
            num_warps=16,
        )
    return grad_input


@triton.jit
def _silu_ppu_flat_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """Flat contiguous FP32 SiLU path tuned for the PPU large-vector case."""
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.fdiv(x, (1.0 + tl.exp(-x)))
    tl.store(out_ptr + offsets, y, mask=mask)


def _try_ppu_flat_silu(inp):
    """Use a measured flat launch for large contiguous PPU FP32 vectors.

    The generic pointwise dispatcher is kept as the default.  This opt-in
    path has deliberately narrow guards so a benchmark can establish an
    end-to-end gain before changing unrelated shapes or dtypes.
    """
    if (
        os.getenv("FLAG_GEMS_PPU_SILU_FAST", "0") != "1"
        or runtime.device.vendor_name != "thead"
        or inp.dtype != torch.float32
        or not inp.is_contiguous()
        or inp.numel() == 0
        or (torch.is_grad_enabled() and inp.requires_grad)
    ):
        return None
    n_elements = int(inp.numel())
    block = 1024 if n_elements >= 1 << 20 else 512
    warps = 8 if n_elements >= 1 << 20 else 4
    output = torch.empty_like(inp)
    with torch_device_fn.device(inp.device):
        _silu_ppu_flat_kernel[(triton.cdiv(n_elements, block),)](
            inp,
            output,
            n_elements,
            BLOCK=block,
            num_warps=warps,
            num_stages=2,
        )
    return output


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    y = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return y


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def silu_backward_kernel(x, dy):
    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    return dx


def silu(self):
    logger.debug("GEMS SILU FORWARD")
    if _can_use_measured_ppu_path(self):
        return _silu_measured_ppu(self)
    candidate = _try_ppu_flat_silu(self)
    if candidate is not None:
        return candidate
    output = silu_forward(self)
    return output


def silu_backward(grad_output, self):
    logger.debug("GEMS SILU BACKWARD")
    if (
        os.getenv("FLAG_GEMS_PPU_SILU_BWD_BLOCK") == "8192"
        and _can_use_measured_ppu_path(self, "FLAG_GEMS_PPU_SILU_BWD_BLOCK")
        and grad_output.is_contiguous()
        and grad_output.shape == self.shape
    ):
        return _silu_backward_measured_ppu(grad_output, self)
    grad_input = silu_backward_kernel(self, grad_output)
    return grad_input


def silu_(A):
    logger.debug("GEMS SILU_ FORWARD")
    if _can_use_measured_ppu_path(A):
        return _silu_measured_ppu(A, out=A)
    if (
        os.getenv("FLAG_GEMS_PPU_SILU_FAST", "0") == "1"
        and runtime.device.vendor_name == "thead"
        and A.dtype == torch.float32
        and A.is_contiguous()
        and A.numel() > 0
        and not (torch.is_grad_enabled() and A.requires_grad)
    ):
        n_elements = int(A.numel())
        block = 1024 if n_elements >= 1 << 20 else 512
        warps = 8 if n_elements >= 1 << 20 else 4
        with torch_device_fn.device(A.device):
            _silu_ppu_flat_kernel[(triton.cdiv(n_elements, block),)](
                A,
                A,
                n_elements,
                BLOCK=block,
                num_warps=warps,
                num_stages=2,
            )
        return A
    out = silu_forward(A, out0=A)
    return out
