import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_scalar_func(inp, value_scalar):
    return tl.full(inp.shape, value_scalar, dtype=inp.dtype)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_tensor_func(inp, value):
    return value


def fill_scalar(input, value):
    logger.debug("GEMS FILL (Dynamic)")
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_scalar_func(input, value, out0=out)


def fill_scalar_out(input, value, *, out=None):
    logger.debug("GEMS FILL_SCALAR_OUT")
    if out is None:
        return fill_scalar(input, value)
    with torch_device_fn.device(input.device):
        fill_scalar_func(input, value, out0=out)
    return out


def fill_tensor(input, value):
    if not value.is_cuda:
        return fill_scalar(input, value.item())
    logger.debug("GEMS FILL (Dynamic)")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_tensor_func(input, value, out0=out)


def fill_tensor_out(input, value, *, out=None):
    logger.debug("GEMS FILL_TENSOR_OUT")
    if out is None:
        return fill_tensor(input, value)
    if not value.is_cuda:
        return fill_scalar_out(input, value.item(), out=out)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    with torch_device_fn.device(input.device):
        fill_tensor_func(input, value, out0=out)
    return out


def fill_tensor_(self, value):
    if not value.is_cuda:
        return fill_scalar_(self, value.item())
    logger.debug("GEMS FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    with torch_device_fn.device(self.device):
        fill_tensor_func(self, value, out0=self)
    return self


@libentry()
@triton.jit(do_not_specialize=["value", "size"])
def _fill_contiguous_fp32(out, value, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + offsets, value, offsets < size)


def fill_scalar_(self, value):
    logger.debug("GEMS FILL_SCALAR_")
    # Flatten contiguous FP32 storage without pointwise broadcasting/codegen.
    # Keep dtype/layout and scalar conversions outside this contract unchanged.
    if self.dtype == torch.float32 and self.is_contiguous() and isinstance(value, (int, float)):
        if not isinstance(value, int) or abs(value) <= (1 << 53):
            scalar = float(value)
            if not math.isfinite(scalar) or abs(scalar) <= 3.4028234663852886e38:
                size = self.numel()
                if size:
                    with torch_device_fn.device(self.device):
                        _fill_contiguous_fp32[(triton.cdiv(size, 1024),)](
                            self, scalar, size, BLOCK=1024, num_warps=4
                        )
                return self
    with torch_device_fn.device(self.device):
        fill_scalar_func(self, value, out0=self)
    return self
