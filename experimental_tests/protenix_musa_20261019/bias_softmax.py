"""MUSA inference: fuse two rounded bias additions and row softmax in place.

The GEMMs, attention scaling, model architecture and generation budget remain
native. Guards retain the original path for unsupported layouts and training.
"""
from collections import Counter
import torch
import triton
import triton.language as tl
from protenix.model.triangular import layers

original = layers._attention
hits = Counter()


@triton.jit
def _bias_softmax(S, B0, B1, H:tl.constexpr, Q:tl.constexpr, K:tl.constexpr,
                 T00:tl.constexpr, T01:tl.constexpr, T02:tl.constexpr, T03:tl.constexpr,
                 T10:tl.constexpr, T11:tl.constexpr, T12:tl.constexpr, T13:tl.constexpr,
                 BLOCK:tl.constexpr):
    # Full attention tensors can exceed 2^31 elements even at N=840.
    # Promote BEFORE multiplying; casting an overflowed offset is too late.
    row = tl.program_id(0).to(tl.int64)
    b = row // (H*Q)
    h = (row // Q) % H
    q = row % Q
    k = tl.arange(0, BLOCK)
    s = tl.load(S+row*K+k, k<K, 0).to(tl.float32)
    b0 = tl.load(B0+b*T00+h*T01+q*T02+k*T03, k<K, 0).to(tl.float32)
    # Native in-place adds round back to score dtype after each addition.
    s = (s+b0).to(S.dtype.element_ty).to(tl.float32)
    b1 = tl.load(B1+b*T10+h*T11+q*T12+k*T13, k<K, 0).to(tl.float32)
    s = (s+b1).to(S.dtype.element_ty).to(tl.float32)
    s = tl.where(k<K, s, -float('inf'))
    e = tl.exp(s-tl.max(s, axis=0))
    p = e/tl.sum(e, axis=0)
    tl.store(S+row*K+k, p, k<K)


def strides(bias, shape):
    if bias.ndim > 4:
        return None
    dims = (1,)*(4-bias.ndim)+tuple(bias.shape)
    steps = (0,)*(4-bias.ndim)+tuple(bias.stride())
    if any(a not in (1, b) for a,b in zip(dims, shape)):
        return None
    return tuple(0 if n == 1 else s for n,s in zip(dims, steps))


def fused_attention(query, key, value, biases):
    if (torch.is_grad_enabled() or query.device.type != 'musa' or query.ndim != 4
            or len(biases) != 2 or query.dtype not in (torch.float32, torch.bfloat16)):
        hits['fallback_guard'] += 1
        return original(query, key, value, biases)
    shape = tuple(query.shape[:-1])+(key.shape[-2],)
    if shape[-1] < 200:
        hits['fallback_small'] += 1
        return original(query, key, value, biases)
    steps = [strides(b, shape) for b in biases]
    if any(s is None for s in steps) or shape[-1] > 4096:
        hits['fallback_layout'] += 1
        return original(query, key, value, biases)
    scores = torch.matmul(query, key.transpose(-1,-2))
    if not scores.is_contiguous():
        hits['fallback_noncontiguous'] += 1
        return original(query, key, value, biases)
    b,h,q,k = scores.shape
    _bias_softmax[(b*h*q,)](scores, biases[0], biases[1], h,q,k,
                            *steps[0], *steps[1], BLOCK=triton.next_power_of_2(k),
                            num_warps=4, enable_fp_fusion=False)
    hits[f'fused_K{k}_{str(scores.dtype)}'] += 1
    return torch.matmul(scores, value)


def install(model=None):
    from attention_compat import wrap
    layers._attention = wrap(fused_attention)
    return 1
