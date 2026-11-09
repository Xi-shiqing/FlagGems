"""PPU SiLU with the validated exponential and division rounding."""

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.ops.silu import silu as _generic_silu
from flag_gems.ops.silu import silu_ as _generic_silu_
from flag_gems.ops.silu import silu_backward as _generic_silu_backward
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.triton_lang_extension import div_rn


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _silu_rn(x):
    x = x.to(tl.float32)
    return div_rn(x, 1.0 + libdevice.exp(-x))


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _silu_backward_rn(x, dy):
    x, dy = x.to(tl.float32), dy.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + libdevice.exp(-x))
    return dy * sigma * (1.0 + x * (1.0 - sigma))


@triton.jit
def _silu_flat(x, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, offsets < n, other=0.0).to(tl.float32)
    result = div_rn(values, 1.0 + libdevice.exp(-values))
    tl.store(out + offsets, result, offsets < n)


def _run(x, out=None):
    if (
        os.environ.get("FLAG_GEMS_PPU_SILU_HOT") == "1"
        and x.is_contiguous()
        and x.numel() > 0
        and not x.requires_grad
    ):
        if out is None:
            out = torch.empty_like(x)
        with torch_device_fn.device(x.device):
            _silu_flat[(triton.cdiv(x.numel(), 4096),)](
                x, out, x.numel(), BLOCK=4096, num_warps=8, num_stages=2
            )
        return out
    return _silu_rn(x) if out is None else _silu_rn(x, out0=out)


def silu(self):
    return _run(self) if self.dtype == torch.float32 else _generic_silu(self)


def silu_(self):
    return _run(self, self) if self.dtype == torch.float32 else _generic_silu_(self)


def silu_backward(grad_output, self):
    if self.dtype == grad_output.dtype == torch.float32:
        return _silu_backward_rn(self, grad_output)
    return _generic_silu_backward(grad_output, self)
