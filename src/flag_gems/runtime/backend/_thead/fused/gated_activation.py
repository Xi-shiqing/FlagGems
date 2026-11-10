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


"""PPU FP32 gating with the rounding of the separate FlagGems operations."""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
from triton.language.extra import libdevice

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import full_write_allocation, libentry, pointwise_dynamic
from flag_gems.utils.triton_lang_extension import div_rn


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _sigmoid_mul(gate, value):
    gate, value = gate.to(tl.float32), value.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + libdevice.exp(-gate))
    return _mul_rn(sigma, value)


@pointwise_dynamic(promotion_methods=[(0, 1, 2, "DEFAULT")])
@triton.jit
def _sigmoid_mul_add(gate, value, shift):
    gate, value, shift = gate.to(tl.float32), value.to(tl.float32), shift.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + libdevice.exp(-gate))
    return _add_rn(_mul_rn(sigma, value), shift)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _silu_mul(gate, value):
    gate, value = gate.to(tl.float32), value.to(tl.float32)
    activation = div_rn(gate, 1.0 + libdevice.exp(-gate))
    return _mul_rn(activation, value)


@pointwise_dynamic(promotion_methods=[(0, 1, 2, "DEFAULT")])
@triton.jit
def _mul_add(activation, value, shift):
    activation, value, shift = activation.to(tl.float32), value.to(tl.float32), shift.to(tl.float32)
    return _add_rn(_mul_rn(activation, value), shift)


@libentry()
@triton.jit
def _flat(gate, value, shift, output, n, MODE: tl.constexpr, HAS_SHIFT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(gate + offsets, mask, other=0.0).to(tl.float32)
    y = tl.load(value + offsets, mask, other=0.0).to(tl.float32)
    if MODE == 0:
        x = div_rn(1.0, 1.0 + libdevice.exp(-x))
    elif MODE == 1:
        x = div_rn(x, 1.0 + libdevice.exp(-x))
    result = _mul_rn(x, y)
    if HAS_SHIFT:
        bias = tl.load(shift + offsets, mask, other=0.0).to(tl.float32)
        result = _add_rn(result, bias)
    tl.store(output + offsets, result, mask)


def _check_inputs(gate, value, shift):
    tensors = [gate, value] + ([] if shift is None else [shift])
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise TypeError("PPU gating requires FP32 inputs")
    if any(tensor.device != gate.device for tensor in tensors) or gate.device.type != "cuda":
        raise ValueError("PPU gating inputs must share one CUDA-compatible device")


def _run(gate, value, shift, mode):
    _check_inputs(gate, value, shift)
    flat = gate.shape == value.shape and gate.is_contiguous() and value.is_contiguous()
    if shift is not None:
        flat = flat and shift.shape == value.shape and shift.is_contiguous()
    if flat:
        output = full_write_allocation.empty_like(value)
        n = value.numel()
        if n:
            block = 2048 if n >= 2**21 else 512
            with torch_device_fn.device(gate.device):
                _flat[((n + block - 1) // block,)](
                    gate, value, gate if shift is None else shift, output, n,
                    MODE=mode, HAS_SHIFT=shift is not None, BLOCK=block,
                    num_warps=8 if block == 2048 else 4,
                )
        return output
    if mode == 1:
        return _silu_mul(gate, value)
    if mode == 2:
        return flag_gems.mul(gate, value) if shift is None else _mul_add(gate, value, shift)
    return _sigmoid_mul(gate, value) if shift is None else _sigmoid_mul_add(gate, value, shift)


def _sum_to_shape(gradient, shape):
    if not shape:
        return flag_gems.sum(gradient)
    leading = gradient.ndim - len(shape)
    axes = list(range(leading))
    axes += [leading + index for index, size in enumerate(shape)
             if size == 1 and gradient.shape[leading + index] != 1]
    if axes:
        gradient = flag_gems.sum_dim(gradient, dim=axes, keepdim=True)
    return gradient.reshape(shape)


class _Gating(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, value, shift, kind, needs_backward):
        _check_inputs(gate, value, shift)
        ctx.kind = kind
        ctx.shapes = (tuple(gate.shape), tuple(value.shape), None if shift is None else tuple(shift.shape))
        ctx.product_shape = tuple(torch.broadcast_shapes(gate.shape, value.shape))
        if needs_backward:
            activation = flag_gems.silu(gate) if kind == "silu" else flag_gems.sigmoid(gate)
            need_gate, need_value = ctx.needs_input_grad[:2]
            ctx.save_for_backward(
                value if need_gate else None,
                activation if need_value or (need_gate and kind != "silu") else None,
                gate if need_gate and kind == "silu" else None,
            )
            return _run(activation, value, shift, 2)
        return _run(gate, value, shift, 1 if kind == "silu" else 0)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, activation, raw_silu = ctx.saved_tensors
        gate_shape, value_shape, shift_shape = ctx.shapes
        grad_gate = grad_value = grad_shift = None
        # Add backward reduces any dimensions introduced by the shift before
        # Mul backward multiplies by its saved inputs.
        grad_product = _sum_to_shape(grad_output, ctx.product_shape) if shift_shape is not None else grad_output
        if ctx.needs_input_grad[0]:
            # The separate Mul backward reduces broadcast dimensions before
            # Sigmoid/SiLU backward. Preserve that ordering and its rounding.
            derivative_input = _sum_to_shape(flag_gems.mul(grad_product, value), gate_shape)
            if ctx.kind == "silu":
                grad_gate = flag_gems.silu_backward(derivative_input, raw_silu)
            else:
                grad_gate = flag_gems.sigmoid_backward(derivative_input, activation)
        if ctx.needs_input_grad[1]:
            grad_value = _sum_to_shape(flag_gems.mul(grad_product, activation), value_shape)
        if ctx.needs_input_grad[2]:
            grad_shift = _sum_to_shape(grad_output, shift_shape)
        return grad_gate, grad_value, grad_shift, None, None


def sigmoid_mul(gate, value):
    needs_backward = torch.is_grad_enabled() and (gate.requires_grad or value.requires_grad)
    if not needs_backward:
        return _run(gate, value, None, 0)
    return _Gating.apply(gate, value, None, "sigmoid", needs_backward)


def sigmoid_mul_add(gate, value, shift):
    needs_backward = torch.is_grad_enabled() and any(tensor.requires_grad for tensor in [gate, value, shift])
    if not needs_backward:
        return _run(gate, value, shift, 0)
    return _Gating.apply(gate, value, shift, "sigmoid", needs_backward)


def silu_mul(gate, value):
    needs_backward = torch.is_grad_enabled() and (gate.requires_grad or value.requires_grad)
    if not needs_backward:
        return _run(gate, value, None, 1)
    return _Gating.apply(gate, value, None, "silu", needs_backward)
