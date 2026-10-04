import logging
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic, tl_extra_shim
from flag_gems.utils.triton_lang_extension import div_rn

logger = logging.getLogger(__name__)
exp2 = tl_extra_shim.exp2


@triton.jit
def _sigmoid_contiguous_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Flat contiguous sigmoid path for the profiled PPU pair-shape family."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = div_rn(1.0, 1.0 + libdevice.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _sigmoid_backward_contiguous_kernel(dy_ptr, y_ptr, dx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Flat sigmoid VJP with the same grouping as ``sigmoid_backward_kernel``."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dx = dy * (1.0 - y) * y
    tl.store(dx_ptr + offsets, dx, mask=mask)


def _can_use_measured_ppu_path(x):
    # The flat path is only for PPU inference (or explicit opt-in).  A raw
    # Triton write is not an autograd operation, so tensors participating in a
    # training graph stay on the pointwise implementation below.
    return (
        (os.getenv("PPU_SDK") or os.getenv("FLAG_GEMS_PPU_SIGMOID_FLAT") == "1")
        and x.device.type == "cuda"
        and x.dtype == torch.float32
        and x.is_contiguous()
        and x.ndim >= 1
        and x.numel() >= (1 << 18)
        and not x.requires_grad
    )


def _sigmoid_measured_ppu(x, out=None):
    if out is None:
        out = torch.empty_like(x)
    n_elements = x.numel()
    # The sweep on the Protenix trace shape (693, 768) selects 512x4.  Keep a
    # larger tile for very large tensors to avoid excessive CTA count.
    block = 512 if n_elements < (1 << 20) else 2048
    warps = 4 if block == 512 else 8
    grid = (triton.cdiv(n_elements, block),)
    with torch_device_fn.device(x.device):
        _sigmoid_contiguous_kernel[grid](
            x, out, n_elements, BLOCK_SIZE=block, num_warps=warps
        )
    return out


def _sigmoid_backward_measured_ppu(grad_output, output):
    if not (
        (os.getenv("PPU_SDK") or os.getenv("FLAG_GEMS_PPU_SIGMOID_FLAT") == "1")
        and not torch.is_grad_enabled()
        and isinstance(grad_output, torch.Tensor)
        and isinstance(output, torch.Tensor)
        and not grad_output.requires_grad
        and grad_output.device == output.device
        and output.device.type == "cuda"
        and grad_output.dtype == torch.float32
        and output.dtype == torch.float32
        and grad_output.is_contiguous()
        and output.is_contiguous()
        and grad_output.shape == output.shape
        and output.ndim >= 1
        and output.numel() >= (1 << 18)
    ):
        return None
    dx = torch.empty_like(output)
    n_elements = output.numel()
    block = 512 if n_elements < (1 << 20) else 2048
    warps = 4 if block == 512 else 8
    with torch_device_fn.device(output.device):
        _sigmoid_backward_contiguous_kernel[(triton.cdiv(n_elements, block),)](
            grad_output, output, dx, n_elements, BLOCK_SIZE=block, num_warps=warps
        )
    return dx


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_forward(x):
    # log2e: tl.constexpr = math.log2(math.e)
    # Triton 3.0.0 disallows calling non-jitted functions inside jitted
    # functions, so the old constexpr spelling is intentionally omitted.
    return div_rn(1.0, 1.0 + libdevice.exp(-x.to(tl.float32)))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_backward_kernel(dy, y):
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    return dy_f32 * (1.0 - y_f32) * y_f32


def sigmoid(self):
    logger.debug("GEMS SIGMOID FORWARD")
    if _can_use_measured_ppu_path(self):
        return _sigmoid_measured_ppu(self)
    output = sigmoid_forward(self)
    return output


def sigmoid_backward(grad_output, output):
    logger.debug("GEMS SIGMOID BACKWARD")
    candidate = _sigmoid_backward_measured_ppu(grad_output, output)
    if candidate is not None:
        return candidate
    return sigmoid_backward_kernel(grad_output, output)


def sigmoid_(A):
    logger.debug("GEMS SIGMOID_ FORWARD")
    if _can_use_measured_ppu_path(A):
        return _sigmoid_measured_ppu(A, out=A)
    out = sigmoid_forward(A, out0=A)
    return out
