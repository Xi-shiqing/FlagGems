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

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)
exp2 = tl_extra_shim.exp2


@triton.jit
def _sigmoid_ppu_flat_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Flat contiguous FP32 sigmoid for the large PPU inference tensors.

    The generic pointwise dispatcher decodes the complete N-D index for every
    element.  These tensors are dense and equal-shaped, so one flat 64-bit
    address calculation is sufficient and avoids that per-element overhead.
    ``exp2`` and the same log2(e) constant are retained so the numerical path
    is identical to :func:`sigmoid_forward`.
    """
    pid = tl.program_id(0).to(tl.int64)
    offsets = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offsets < n_elements
    log2e: tl.constexpr = 1.4426950408889634
    x_f32 = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / (1.0 + exp2(-x_f32 * log2e))
    tl.store(out_ptr + offsets, y, mask=mask)


def _try_ppu_flat_sigmoid(self):
    """Launch the measured dense PPU schedule when it is safe to do so.

    This is intentionally narrow: FP32, contiguous, large, inference-only
    inputs.  The established pointwise implementation remains the fallback
    for views, other dtypes, small tensors, and autograd/training.
    """
    if (
        os.getenv(
            "FLAG_GEMS_PPU_SIGMOID_FAST",
            # The flat kernel wins the isolated microbench, but the same-input
            # Protenix end-to-end A/B is neutral/slightly slower (0.067%).
            # Keep it opt-in until a workload where sigmoid is on the critical
            # path demonstrates a reproducible wall-time gain.
            "0",
        )
        != "1"
        or runtime_device.vendor_name != "thead"
        or self.dtype != torch.float32
        or not self.is_contiguous()
        or self.numel() < (1 << 20)
        or (torch.is_grad_enabled() and self.requires_grad)
    ):
        return None
    n_elements = int(self.numel())
    output = torch.empty_like(self)
    block = 8192 if n_elements >= (1 << 24) else 2048
    warps = 8 if n_elements >= (1 << 24) else 4
    with torch_device_fn.device(self.device):
        _sigmoid_ppu_flat_kernel[(triton.cdiv(n_elements, block),)](
            self,
            output,
            n_elements,
            BLOCK_SIZE=block,
            num_warps=warps,
            num_stages=2,
        )
    return output


def _try_ppu_flat_sigmoid_inplace(self):
    """In-place counterpart for the large dense inference tensors."""
    if (
        os.getenv(
            "FLAG_GEMS_PPU_SIGMOID_FAST",
            "0",
        )
        != "1"
        or runtime_device.vendor_name != "thead"
        or self.dtype != torch.float32
        or not self.is_contiguous()
        or self.numel() < (1 << 20)
        or (torch.is_grad_enabled() and self.requires_grad)
    ):
        return False
    n_elements = int(self.numel())
    block = 8192 if n_elements >= (1 << 24) else 2048
    warps = 8 if n_elements >= (1 << 24) else 4
    with torch_device_fn.device(self.device):
        _sigmoid_ppu_flat_kernel[(triton.cdiv(n_elements, block),)](
            self,
            self,
            n_elements,
            BLOCK_SIZE=block,
            num_warps=warps,
            num_stages=2,
        )
    return True


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_forward(x):
    # log2e: tl.constexpr = math.log2(math.e)
    # triton 3.0.0 disallow calling non-jitted function inside jitted function, even if it is in
    # the rhs of an assignment to a constexpr, so we use numeric literal instead to work around this.
    log2e: tl.constexpr = 1.4426950408889634
    return 1 / (1 + exp2(-x.to(tl.float32) * log2e))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_backward_kernel(dy, y):
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    return dy_f32 * (1.0 - y_f32) * y_f32


def sigmoid(self):
    logger.debug("GEMS SIGMOID FORWARD")
    fast_output = _try_ppu_flat_sigmoid(self)
    if fast_output is not None:
        return fast_output
    output = sigmoid_forward(self)
    return output


def sigmoid_backward(grad_output, output):
    logger.debug("GEMS SIGMOID BACKWARD")
    grad_input = sigmoid_backward_kernel(grad_output, output)
    return grad_input


def sigmoid_(A):
    logger.debug("GEMS SIGMOID_ FORWARD")
    if _try_ppu_flat_sigmoid_inplace(A):
        return A
    out = sigmoid_forward(A, out0=A)
    return out
