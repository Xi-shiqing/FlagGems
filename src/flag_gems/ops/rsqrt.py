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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def rsqrt_func(x):
    return tl.rsqrt(x.to(tl.float32))


class _RsqrtExactBackward(torch.autograd.Function):
    """Keep the native FP32 derivative grouping when training."""

    @staticmethod
    def forward(ctx, inp):
        out = rsqrt_func(inp)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        # Evaluate y^3 first, then apply -0.5 and grad_output.  This is the
        # order used by the native PPU FP32 rsqrt backward path.
        return grad_output * (-0.5 * (out * out * out))


def rsqrt(A):
    logger.debug("GEMS RSQRT")
    if A.requires_grad:
        return _RsqrtExactBackward.apply(A)
    return rsqrt_func(A)


def rsqrt_(A):
    logger.debug("GEMS RSQRT_")
    return rsqrt_func(A, out0=A)
