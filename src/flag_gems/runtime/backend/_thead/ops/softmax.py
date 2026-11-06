"""Preserve accepted PPU FP32 Softmax; retain generic dtype and empty handling."""
import torch

from flag_gems.ops.copy import copy_
from flag_gems.ops.softmax import softmax as _generic_softmax
from flag_gems.ops.softmax import softmax_backward as _generic_backward
from flag_gems.ops.softmax import softmax_backward_out as _generic_backward_out
from flag_gems.ops.softmax import softmax_out as _generic_softmax_out

from .softmax_fp32 import softmax as _current_softmax
from .softmax_fp32 import softmax_backward as _current_backward


def _use_current(value):
    return value.dtype == torch.float32 and value.numel() > 0


def softmax(self, dim, half_to_float=False):
    if _use_current(self):
        return _current_softmax(self, dim, half_to_float)
    return _generic_softmax(self, dim, half_to_float)


def softmax_out(self, dim, half_to_float=False, *, out):
    if not _use_current(self):
        return _generic_softmax_out(self, dim, half_to_float, out=out)
    result = _current_softmax(self, dim, half_to_float)
    if tuple(out.shape) != tuple(result.shape):
        out.resize_(result.shape)
    if out.dtype != result.dtype:
        raise RuntimeError("_softmax.out dtype mismatch")
    return copy_(out, result)


def softmax_backward(grad_output, output, dim, input_dtype):
    if _use_current(output):
        return _current_backward(grad_output, output, dim, input_dtype)
    return _generic_backward(grad_output, output, dim, input_dtype)


def softmax_backward_out(grad_output, output, dim, input_dtype, *, grad_input):
    if not _use_current(output):
        return _generic_backward_out(grad_output, output, dim, input_dtype, grad_input=grad_input)
    result = _current_backward(grad_output, output, dim, input_dtype)
    if tuple(grad_input.shape) != tuple(result.shape):
        grad_input.resize_(result.shape)
    if grad_input.dtype != result.dtype:
        raise RuntimeError("_softmax_backward_data.out dtype mismatch")
    return copy_(grad_input, result)
