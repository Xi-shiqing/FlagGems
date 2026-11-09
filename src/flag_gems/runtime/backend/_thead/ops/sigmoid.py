"""Preserve the PPU FP32 sigmoid rounding used by the validated model path."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.ops.sigmoid import sigmoid as _generic_sigmoid
from flag_gems.ops.sigmoid import sigmoid_ as _generic_sigmoid_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import full_write_allocation, pointwise_dynamic
from flag_gems.utils.triton_lang_extension import div_rn


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def _sigmoid_rn(x):
    return div_rn(1.0, 1.0 + libdevice.exp(-x.to(tl.float32)))


@triton.jit
def _sigmoid_contiguous(x, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, mask=offsets < n, other=0.0).to(tl.float32)
    result = div_rn(1.0, 1.0 + libdevice.exp(-values))
    tl.store(out + offsets, result, mask=offsets < n)


def _run(x, out=None):
    if x.is_contiguous() and x.ndim >= 1 and x.numel() >= (1 << 18) and not x.requires_grad:
        if out is None:
            out = full_write_allocation.empty_like(x)
        n = x.numel()
        block = 512 if n < (1 << 20) else 2048
        with torch_device_fn.device(x.device):
            _sigmoid_contiguous[(triton.cdiv(n, block),)](
                x, out, n, BLOCK=block, num_warps=4 if block == 512 else 8
            )
        return out
    if out is None:
        return _sigmoid_rn(x)
    return _sigmoid_rn(x, out0=out)


def sigmoid(self):
    if self.dtype == torch.float32:
        return _run(self)
    return _generic_sigmoid(self)


def sigmoid_(self):
    if self.dtype == torch.float32:
        return _run(self, self)
    return _generic_sigmoid_(self)
