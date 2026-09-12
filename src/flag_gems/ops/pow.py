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

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS POW_TENSOR_TENSOR")
    return pow_func(A, exponent)


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS POW_TENSOR_TENSOR_")
    return pow_func(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def pow_square_fp32_rn_func(x):
    """Compute the common FP32 square with one explicitly rounded multiply.

    The generic backend ``pow`` path lowers through the transcendental
    implementation even when the exponent is exactly two.  Protenix uses
    this spelling in its distance feature path; using one FP32 multiply keeps
    the operation's rounding point aligned with the native ``x * x`` path.
    """
    return tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


def _is_fp32_square(A, exponent):
    return A.dtype == torch.float32 and exponent == 2


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS POW_TENSOR_SCALAR")
    if _is_fp32_square(A, exponent):
        return pow_square_fp32_rn_func(A)
    return pow_func_tensor_scalar(A, exponent)


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS POW_TENSOR_SCALAR_")
    if _is_fp32_square(A, exponent):
        return pow_square_fp32_rn_func(A, out0=A)
    return pow_func_tensor_scalar(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_scalar(A, exponent):
    logger.debug("GEMS POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)
