"""Pack strided operands once and retain the validated BMM reduction kernels."""
import math

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.ops._scaled_dot_product_attention_math import (
    _scaled_dot_product_attention_math as _generic_sdpa,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import native_output_allocation


@libentry()
@triton.jit
def _pack(
    Q, K, V, QP, KP, VP, BATCH: tl.constexpr, HEADS: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
    Q_SCALE: tl.constexpr, K_SCALE: tl.constexpr, BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    qh, kh, vh = i // (M * D), i // (N * D), i // (N * DV)
    qm, kn, vn = i // D % M, i // D % N, i // DV % N
    q = tl.load(Q + qh // HEADS * QS[0] + qh % HEADS * QS[1] + qm * QS[2] + i % D * QS[3], i < BATCH * HEADS * M * D, 0.0)
    k = tl.load(K + kh // HEADS * KS[0] + kh % HEADS * KS[1] + kn * KS[2] + i % D * KS[3], i < BATCH * HEADS * N * D, 0.0)
    v = tl.load(V + vh // HEADS * VS[0] + vh % HEADS * VS[1] + vn * VS[2] + i % DV * VS[3], i < BATCH * HEADS * N * DV, 0.0)
    tl.store(QP + i, q * Q_SCALE, i < BATCH * HEADS * M * D)
    tl.store(KP + i, k * K_SCALE, i < BATCH * HEADS * N * D)
    tl.store(VP + i, v, i < BATCH * HEADS * N * DV)


@triton.jit
def _bias_softmax(
    S, MASK, P, HEADS: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
    MS: tl.constexpr, HAS_MASK: tl.constexpr, BOOL_MASK: tl.constexpr,
    CAUSAL: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    values = tl.load(S + row * N + cols, cols < N, other=-float("inf"))
    if HAS_MASK:
        batch_head, m = row // M, row % M
        bias = tl.load(MASK + batch_head // HEADS * MS[0] + batch_head % HEADS * MS[1] + m * MS[2] + cols * MS[3], cols < N, 0.0)
        if BOOL_MASK:
            values = tl.where(bias, values, -float("inf"))
        else:
            values = values + bias
    if CAUSAL:
        values = tl.where(row % M >= cols, values, -float("inf"))
    maximum = tl.max(values, axis=0)
    exp_values = tl.exp(values - maximum)
    probabilities = exp_values / tl.sum(exp_values, axis=0)
    probabilities = tl.where(maximum == -float("inf"), 0.0, probabilities)
    tl.store(P + row * N + cols, probabilities, cols < N)


def _ordered_forward(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    assert dropout_p == 0 and query.dtype == key.dtype == value.dtype == torch.float32
    assert query.ndim == key.ndim == value.ndim and query.ndim in (3, 4)
    batch, heads, rows, dim = query.shape if query.ndim == 4 else (1, *query.shape)
    cols, value_dim = value.shape[-2:]
    assert 0 < cols <= 4096
    assert key.shape[-2:] == (cols, dim)
    assert key.shape[:-2] == query.shape[:-2] == value.shape[:-2]
    scale = 1.0 / math.sqrt(dim) if scale is None else scale
    factor = math.sqrt(abs(scale))
    strides = lambda x: x.stride() if x.ndim == 4 else (0, *x.stride())
    allocate = native_output_allocation.empty_for_overwrite
    q = allocate((batch * heads, rows, dim), device=query.device, dtype=query.dtype)
    k = allocate((batch * heads, cols, dim), device=query.device, dtype=query.dtype)
    v = allocate((batch * heads, cols, value_dim), device=query.device, dtype=query.dtype)
    weights = allocate((batch * heads, rows, cols), device=query.device, dtype=query.dtype)
    mask = attn_mask.expand((*query.shape[:-2], rows, cols)) if attn_mask is not None else query
    with torch_device_fn.device(query.device):
        _pack[(triton.cdiv(max(q.numel(), k.numel(), v.numel()), 1024),)](
            query, key, value, q, k, v, batch, heads, rows, cols, dim, value_dim,
            strides(query), strides(key), strides(value), factor if scale >= 0 else -factor, factor,
            BLOCK=1024, num_warps=4,
        )
        scores = flag_gems.bmm(q, k.transpose(1, 2))
        _bias_softmax[(batch * heads * rows,)](
            scores, mask, weights, heads, rows, cols, strides(mask) if attn_mask is not None else (0, 0, 0, 0),
            attn_mask is not None, attn_mask is not None and attn_mask.dtype == torch.bool, is_causal,
            BLOCK=triton.next_power_of_2(cols),
        )
        output = flag_gems.bmm(weights, v)
    return output.view(*query.shape[:-1], value_dim), weights.view(*query.shape[:-2], rows, cols)


def _scaled_dot_product_attention_math(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    tensors = (query, key, value) + (() if attn_mask is None else (attn_mask,))
    use_ordered = (
        dropout_p == 0.0
        and query.dtype == key.dtype == value.dtype == torch.float32
        and query.ndim in (3, 4)
        and query.ndim == key.ndim == value.ndim
        and query.shape[:-2] == key.shape[:-2] == value.shape[:-2]
        and query.shape[-1] == key.shape[-1]
        and key.shape[-2] == value.shape[-2]
        and query.numel() > 0
        and value.shape[-1] > 0
        # A complete row fits the same single-program softmax as the
        # established math path. This bound is independent of model/input.
        and 0 < key.shape[-2] <= 4096
        and (attn_mask is None or attn_mask.dtype in (torch.float32, torch.bool))
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))
    )
    if use_ordered:
        return _ordered_forward(query, key, value, attn_mask, dropout_p, is_causal, scale)
    return _generic_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale)
