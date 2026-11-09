"""Overlap long-K input loads while preserving the established accumulation."""
import torch
import triton

from flag_gems.ops.tf32_precision import empty_full_write
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.device_info import get_sm_count


def try_mm(a, b, out, kernel, tf32):
    if (
        not a.is_cuda or not b.is_cuda or a.device != b.device
        or a.dtype != torch.float32 or b.dtype != torch.float32
        or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]
        or a.is_neg() or b.is_neg()
    ):
        return None
    m, k = a.shape
    n = b.shape[1]
    if (
        k <= 8192 or not (32 <= m <= 1024 and 32 <= n <= 1024)
        or a.stride() != (1, m) or b.stride() != (n, 1)
    ):
        return None
    tasks = triton.cdiv(m, 32) * triton.cdiv(n, 32)
    with torch_device_fn.device(a.device):
        # At most four output waves: extra load overlap is useful while the
        # output itself supplies too few blocks to hide the long dependency.
        if tasks > 4 * get_sm_count():
            return None
        result = out if out is not None else empty_full_write(
            (m, n), dtype=a.dtype, device=a.device
        )
        kernel[(tasks,)](
            a, b, result, m, n, k, *a.stride(), *b.stride(), *result.stride(),
            SPLIT_K=1, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
            ALLOW_TF32=tf32, B_RNE_PREPACKED=False, INDEX_64=True,
            num_warps=4, num_stages=3,
        )
    return result
