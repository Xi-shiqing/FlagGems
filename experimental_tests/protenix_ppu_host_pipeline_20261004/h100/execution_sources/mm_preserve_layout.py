"""Round a reused RHS once without changing its physical order or MM reduction."""
from collections import Counter
import torch
import triton
import triton.language as tl
from flag_gems.ops.tf32_precision import empty_full_write, round_tf32_rne

AUDIT = False
CALLS = Counter()


@triton.jit
def _round_rhs(B, P, K: tl.constexpr, N: tl.constexpr,
               COL_MAJOR: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    # Dense physical order is preserved, so these addresses are contiguous.
    value = tl.load(B + offset, offset < K*N, other=0)
    tl.store(P + offset, round_tf32_rne(value, True), offset < K*N)


def make_direct(module):
    original = module._direct_mm
    def _direct_mm(a, b, c, M, N, K):
        allow_tf32 = module._allow_tf32()
        col_major = b.stride() == (1, K)
        row_major = b.stride() == (N, 1)
        # Resource rule: enough M tiles reuse a small dense RHS to amortize
        # one preparation launch. No model/input/shape table or weight cache.
        if (not allow_tf32 or torch.is_grad_enabled() or
                triton.cdiv(M, 64) < 32 or b.numel()*b.element_size() > (64 << 20) or
                not (col_major or row_major)):
            return original(a, b, c, M, N, K)
        packed = empty_full_write((N, K) if col_major else (K, N), dtype=b.dtype, device=b.device)
        if col_major:
            packed = packed.T
        split_k = module._ppu_h100_splitk_parts(M, N, K) or 1
        am, ak = a.stride()
        bk, bn = packed.stride()
        cm, cn = c.stride()
        with module.torch_device_fn.device(a.device):
            _round_rhs[(triton.cdiv(K*N, 1024),)](b, packed, K, N, col_major, 1024,
                                                num_warps=4, num_stages=1)
            module.mm_direct_fused[(triton.cdiv(M, 64)*triton.cdiv(N, 128),)](
                a, packed, c, M, N, K, am, ak, bk, bn, cm, cn,
                SPLIT_K=split_k, BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
                ALLOW_TF32=True, B_RNE_PREPACKED=True,
                INDEX_64=not module._direct_offsets_fit_i32(M,N,K,am,ak,bk,bn,cm,cn,split_k),
                num_warps=4, num_stages=1)
        if AUDIT:
            CALLS['prepared_calls'] += 1
            CALLS['prepared_bytes'] += packed.numel()*packed.element_size()
        return c
    return _direct_mm
