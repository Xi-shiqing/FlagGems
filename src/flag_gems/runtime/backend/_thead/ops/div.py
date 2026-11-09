"""Match the established PPU division rules for scalar and tensor operands."""

import math
import numbers

import torch
import triton

from flag_gems.ops.div import true_div_func_fp32_rn
from flag_gems.ops.div import true_div_func_scalar_tensor_fp32_rn
from flag_gems.ops.div import true_divide as _generic_true_divide
from flag_gems.ops.div import true_divide_out as _generic_true_divide_out
from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def _multiply_reciprocal(x, reciprocal):
    return x * reciprocal


def _tensor_scalar_divide(x, scalar, **kwargs):
    # Native GPU division computes the reciprocal before narrowing a Python
    # scalar to the tensor dtype. Narrowing the divisor first changes -1.7,
    # for example, by one representable FP32 reciprocal value.
    reciprocal = 1.0 / scalar if scalar != 0 else math.copysign(float("inf"), scalar)
    return _multiply_reciprocal(x, reciprocal, **kwargs)


def _function(a, b):
    if isinstance(a, torch.Tensor) and a.dtype == torch.float32:
        if isinstance(b, torch.Tensor) and b.dtype == torch.float32:
            return true_div_func_fp32_rn
        if isinstance(b, numbers.Real):
            # The native GPU scalar-divisor path multiplies by a reciprocal.
            # Replacing it with a rounded division changes attention scaling.
            return _tensor_scalar_divide
    if isinstance(a, numbers.Real) and isinstance(b, torch.Tensor) and b.dtype == torch.float32:
        return true_div_func_scalar_tensor_fp32_rn
    return None


def true_divide(A, B):
    function = _function(A, B)
    return function(A, B) if function is not None else _generic_true_divide(A, B)


def true_divide_out(A, B, out):
    function = _function(A, B)
    return function(A, B, out0=out) if function is not None else _generic_true_divide_out(A, B, out)
