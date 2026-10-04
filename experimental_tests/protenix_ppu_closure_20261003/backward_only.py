"""Install the bounded TF32 backward cache while preserving public forward."""
import importlib

import torch

import combined_64m as cached

public = importlib.import_module('flag_gems.ops.triangle_attention')
triangle_attention = public.triangle_attention
original_backward = public._triangle_attention_backward
FORWARD_FUNCTION_IDENTITY_PRESERVED = triangle_attention is public.triangle_attention


def backward(q, k, v, bias1, bias2, out, lse, do, precision):
    if q.dtype == torch.float32 and precision in {'tf32', 'tf32_rne'}:
        return cached._triangle_attention_backward(q, k, v, bias1, bias2, out, lse, do, precision)
    return original_backward(q, k, v, bias1, bias2, out, lse, do, precision)


public._triangle_attention_backward = backward


def __getattr__(name):
    if name == '_PHYSICAL_BACKWARD_CALLS':
        return cached._PHYSICAL_BACKWARD_CALLS
    raise AttributeError(name)
