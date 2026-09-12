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
import triton.language.extra.libdevice as libdevice

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# The generic Triton ``sqrt`` lowers to the target's fast approximation.  On
# PPU that approximation is not the same rounding path as the native torch
# square root, and it changes Adam updates after the first non-zero step.  Use
# the backend libdevice's round-to-nearest entry point by default on PPU, where
# it matches the H100/native reference.  The explicit environment switch is
# retained for controlled ablations and for non-PPU environments.
_USE_PRECISE_SQRT = os.getenv(
    "FLAG_GEMS_PPU_PRECISE_SQRT",
    "1" if os.getenv("PPU_SDK") else "0",
) == "1"

# The generic pointwise wrapper is useful for arbitrary layouts, but it adds
# indexing/dispatch work even for the overwhelmingly common contiguous FP32
# case.  This second switch is deliberately opt-in until an end-to-end A/B has
# confirmed that the launch overhead is amortised by the real workload.
_USE_PRECISE_FLAT_SQRT = os.getenv("FLAG_GEMS_PPU_PRECISE_SQRT_FLAT", "0") == "1"


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sqrt_func(x):
    return tl.sqrt(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sqrt_precise_func(x):
    return libdevice.sqrt_rn(x.to(tl.float32))


@triton.jit
def sqrt_precise_flat_kernel(
    inp,
    out,
    n,
    BLOCK: tl.constexpr,
):
    """Exact contiguous FP32 sqrt with one flat launch grid.

    ``libdevice.sqrt_rn`` is the PPU path that matches native H100 FP32
    ``torch.sqrt`` byte-for-byte.  Restricting this kernel to contiguous
    tensors keeps pointer arithmetic simple and leaves all broadcast/strided
    cases on the generic implementation above.
    """

    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
    result = libdevice.sqrt_rn(values)
    tl.store(out + offsets, result, mask=mask)


def _precise_flat_sqrt_or_none(inp, out=None):
    if (
        not _USE_PRECISE_FLAT_SQRT
        or runtime.device.vendor_name != "thead"
        or inp.dtype != torch.float32
        or not inp.is_contiguous()
        or inp.numel() == 0
    ):
        return None
    if out is None:
        out = torch.empty_like(inp)
    # 2K gives the best stable trade-off for the large Adam vectors while 1K
    # avoids overprovisioning the tiny tensor cases.
    block = 2048 if inp.numel() >= (1 << 20) else 1024
    grid = (triton.cdiv(inp.numel(), block),)
    with torch_device_fn.device(inp.device):
        sqrt_precise_flat_kernel[grid](
            inp,
            out,
            inp.numel(),
            BLOCK=block,
            num_warps=8,
        )
    return out


def sqrt(A):
    logger.debug("GEMS SQRT")
    use_precise = _USE_PRECISE_SQRT and A.dtype == torch.float32
    if use_precise:
        flat = _precise_flat_sqrt_or_none(A)
        if flat is not None:
            return flat
    return (sqrt_precise_func if use_precise else sqrt_func)(A)


def sqrt_(A):
    logger.debug("GEMS SQRT_")
    use_precise = _USE_PRECISE_SQRT and A.dtype == torch.float32
    if use_precise and _precise_flat_sqrt_or_none(A, out=A) is not None:
        return A
    (sqrt_precise_func if use_precise else sqrt_func)(A, out0=A)
    return A
