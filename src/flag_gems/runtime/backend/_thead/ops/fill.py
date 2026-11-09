"""Low-overhead scalar fill for contiguous FP32 PPU tensors."""
import math
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops.fill import fill_scalar_ as _generic_fill_scalar_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit(do_not_specialize=["value", "n_elements"])
def _fill_contiguous(out, value, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + offsets, value, mask=offsets < n_elements)


def fill_scalar_(self, value):
    if (
        os.getenv("FLAG_GEMS_PPU_CONTIGUOUS_FILL", "1") == "0"
        or not self.is_cuda
        or self.dtype is not torch.float32
        or not self.is_contiguous()
        or self.is_neg()
        or self.is_conj()
    ):
        return _generic_fill_scalar_(self, value)
    # Converting these integers to double is exact and preserves their final
    # FP32 rounding. Larger integers and complex scalars keep the generic path.
    if isinstance(value, int) and -(2**53) <= value <= 2**53:
        value = float(value)
    if not isinstance(value, float):
        return _generic_fill_scalar_(self, value)
    if math.isfinite(value) and abs(value) > 3.4028234663852886e38:
        raise RuntimeError("value cannot be converted to type float without overflow")
    n = self.numel()
    if n == 0:
        return self
    with torch_device_fn.device(self.device):
        _fill_contiguous[(triton.cdiv(n, 1024),)](
            self, value, n, BLOCK=1024, num_warps=4, num_stages=1
        )
    return self
