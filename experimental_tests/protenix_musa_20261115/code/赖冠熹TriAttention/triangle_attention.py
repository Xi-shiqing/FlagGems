# Copyright 2026 FlagGems Triangle Attention contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Streaming Triangle Attention kernels for CUDA-compatible FlagOS devices.

This implementation performs online softmax and never materializes the
S-by-S attention matrix.  Training saves only one log-sum-exp value per query
row, then recomputes attention tiles in backward.  The kernel accepts the
BNSHD layout with FP32/BF16/FP16 inputs, D16/D32/D64, and arbitrary
positive B/N/S/H dimensions. Enabling this backend is strict: unsupported
inputs must fail explicitly and must never silently execute another kernel.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


_PHYSICAL_FORWARD_CALLS = 0
_PHYSICAL_BACKWARD_CALLS = 0
_SHAPE_COUNTS: dict[str, int] = {}
_BACKWARD_CONFIG_COUNTS: dict[str, int] = {}
_FORWARD_PRECISION_COUNTS: dict[str, int] = {}

__operator_version__ = "0.1.0"

_PRECISION_TO_INTERNAL = {
    "ieee": "none",
    "tf32": "full",
    "qk_tf32": "qk",
    "pv_tf32": "pv",
    "tf32x3": "x3",
    "qk_tf32x3": "qkx3",
    "pv_tf32x3": "pvx3",
}
_LAYOUTS = {"BNSHD", "BNHSD"}


def _write_trace() -> None:
    trace_path = os.getenv("FLAG_GEMS_TRIANGLE_ATTENTION_TRACE")
    if not trace_path:
        return
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "physical_forward_calls": _PHYSICAL_FORWARD_CALLS,
                "physical_backward_calls": _PHYSICAL_BACKWARD_CALLS,
                "shapes": _SHAPE_COUNTS,
                "forward_precision_modes": _FORWARD_PRECISION_COUNTS,
                "backward_configs": _BACKWARD_CONFIG_COUNTS,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


atexit.register(_write_trace)


@triton.jit
def _streaming_triangle_attention_fwd_ieee_v1(
    q_ptr,
    k_ptr,
    v_ptr,
    bias1_ptr,
    bias2_ptr,
    out_ptr,
    lse_ptr,
    n_size: tl.constexpr,
    s_size: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    save_lse: tl.constexpr,
    bias_seeded_qk: tl.constexpr,
    qk_allow_tf32: tl.constexpr,
    pv_allow_tf32: tl.constexpr,
):
    start_m = tl.program_id(0) * block_m
    h = tl.program_id(1)
    bn = tl.program_id(2)
    b = bn // n_size
    n = bn % n_size

    offs_m = start_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, head_dim)
    valid_m = offs_m < s_size
    base = ((b * n_size + n) * s_size * n_heads + h) * head_dim
    q_offsets = base + offs_m[:, None] * n_heads * head_dim + offs_d[None, :]
    q = tl.load(q_ptr + q_offsets, mask=valid_m[:, None], other=0.0)

    input_dtype = q_ptr.dtype.element_ty
    qk_precision: tl.constexpr = (
        "bf16x6"
        if qk_allow_tf32 == 4
        else (
            "bf16x3"
            if qk_allow_tf32 == 3
            else (
                "tf32x3"
                if qk_allow_tf32 == 2
                else ("tf32" if qk_allow_tf32 else "ieee")
            )
        )
    )
    pv_precision: tl.constexpr = (
        "bf16x6"
        if pv_allow_tf32 == 4
        else (
            "bf16x3"
            if pv_allow_tf32 == 3
            else (
                "tf32x3"
                if pv_allow_tf32 == 2
                else ("tf32" if pv_allow_tf32 else "ieee")
            )
        )
    )
    m_i = tl.full((block_m,), -float("inf"), tl.float32)
    if bias_seeded_qk:
        l_i = tl.full((block_m,), 1.0, tl.float32)
    else:
        l_i = tl.zeros((block_m,), tl.float32)
    acc = tl.zeros((block_m, head_dim), tl.float32)

    for block_index in range(0, tl.cdiv(s_size, block_n)):
        offs_n = block_index * block_n + tl.arange(0, block_n)
        valid_n = offs_n < s_size
        kv_offsets = base + offs_n[:, None] * n_heads * head_dim + offs_d[None, :]
        k = tl.load(k_ptr + kv_offsets, mask=valid_n[:, None], other=0.0)
        v = tl.load(v_ptr + kv_offsets, mask=valid_n[:, None], other=0.0)

        bias1_base = (b * n_size + n) * s_size
        bias1 = tl.load(
            bias1_ptr + bias1_base + offs_n[None, :],
            mask=valid_n[None, :],
            other=-1.0e9,
        )
        bias2_base = (b * n_heads + h) * s_size * s_size
        bias2_offsets = bias2_base + offs_m[:, None] * s_size + offs_n[None, :]
        valid_scores = valid_m[:, None] & valid_n[None, :]
        bias2 = tl.load(
            bias2_ptr + bias2_offsets,
            mask=valid_scores,
            other=0.0,
        )
        if bias_seeded_qk:
            # The biases seed the FP32 dot accumulator. This preserves the
            # accumulation order used by the tuned Hopper IEEE path.
            scores = bias1.to(tl.float32) + bias2.to(tl.float32)
            scores = tl.dot(
                q,
                tl.trans(k),
                scores,
                input_precision=qk_precision,
            )
        else:
            scores = tl.dot(q, tl.trans(k), input_precision=qk_precision)
            scores = scores + bias1 + bias2
        scores = scores * 1.4426950408889634
        scores = tl.where(valid_scores, scores, -1.0e9)

        m_ij = tl.maximum(tl.max(scores, axis=1), m_i)
        alpha = tl.exp2(m_i - m_ij)
        p = tl.exp2(scores - m_ij[:, None])
        p = tl.where(valid_scores, p, 0.0)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(input_dtype), v, acc, input_precision=pv_precision)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    safe_l = tl.where(valid_m, l_i, 1.0)
    out = acc / safe_l[:, None]
    tl.store(out_ptr + q_offsets, out, mask=valid_m[:, None])
    if save_lse:
        lse_offsets = ((b * n_size + n) * s_size + offs_m) * n_heads + h
        lse = m_i + tl.math.log2(safe_l)
        tl.store(lse_ptr + lse_offsets, lse, mask=valid_m)


@triton.jit
def _streaming_triangle_attention_bwd_preprocess_v1(
    out_ptr,
    do_ptr,
    delta_ptr,
    n_size: tl.constexpr,
    s_size: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
):
    start_m = tl.program_id(0) * block_m
    h = tl.program_id(1)
    bn = tl.program_id(2)
    b = bn // n_size
    n = bn % n_size

    offs_m = start_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, head_dim)
    valid_m = offs_m < s_size
    base = ((b * n_size + n) * s_size * n_heads + h) * head_dim
    offsets = base + offs_m[:, None] * n_heads * head_dim + offs_d[None, :]
    out = tl.load(out_ptr + offsets, mask=valid_m[:, None], other=0.0)
    do = tl.load(do_ptr + offsets, mask=valid_m[:, None], other=0.0)
    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
    delta_offsets = ((b * n_size + n) * s_size + offs_m) * n_heads + h
    tl.store(delta_ptr + delta_offsets, delta, mask=valid_m)


@triton.jit
def _streaming_triangle_attention_bwd_dq_v1(
    q_ptr,
    k_ptr,
    v_ptr,
    bias1_ptr,
    bias2_ptr,
    lse_ptr,
    delta_ptr,
    do_ptr,
    dq_ptr,
    n_size: tl.constexpr,
    s_size: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    qk_precision_mode: tl.constexpr,
    pv_precision_mode: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty
    qk_precision: tl.constexpr = (
        "tf32x3"
        if qk_precision_mode == 2
        else ("tf32" if qk_precision_mode else "ieee")
    )
    pv_precision: tl.constexpr = (
        "tf32x3"
        if pv_precision_mode == 2
        else ("tf32" if pv_precision_mode else "ieee")
    )
    start_m = tl.program_id(0) * block_m
    h = tl.program_id(1)
    bn = tl.program_id(2)
    b = bn // n_size
    n = bn % n_size

    offs_m = start_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, head_dim)
    valid_m = offs_m < s_size
    base = ((b * n_size + n) * s_size * n_heads + h) * head_dim
    q_offsets = base + offs_m[:, None] * n_heads * head_dim + offs_d[None, :]
    q = tl.load(q_ptr + q_offsets, mask=valid_m[:, None], other=0.0)
    do = tl.load(do_ptr + q_offsets, mask=valid_m[:, None], other=0.0)
    row_offsets = ((b * n_size + n) * s_size + offs_m) * n_heads + h
    lse = tl.load(lse_ptr + row_offsets, mask=valid_m, other=0.0)
    delta = tl.load(delta_ptr + row_offsets, mask=valid_m, other=0.0)
    dq = tl.zeros((block_m, head_dim), tl.float32)

    for block_index in range(0, tl.cdiv(s_size, block_n)):
        offs_n = block_index * block_n + tl.arange(0, block_n)
        valid_n = offs_n < s_size
        kv_offsets = base + offs_n[:, None] * n_heads * head_dim + offs_d[None, :]
        k = tl.load(k_ptr + kv_offsets, mask=valid_n[:, None], other=0.0)
        v = tl.load(v_ptr + kv_offsets, mask=valid_n[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision=qk_precision)
        bias1_base = (b * n_size + n) * s_size
        bias1 = tl.load(
            bias1_ptr + bias1_base + offs_n[None, :],
            mask=valid_n[None, :],
            other=-1.0e9,
        )
        bias2_base = (b * n_heads + h) * s_size * s_size
        bias2_offsets = bias2_base + offs_m[:, None] * s_size + offs_n[None, :]
        valid_scores = valid_m[:, None] & valid_n[None, :]
        bias2 = tl.load(bias2_ptr + bias2_offsets, mask=valid_scores, other=0.0)
        scores = (scores + bias1 + bias2) * 1.4426950408889634
        p = tl.exp2(scores - lse[:, None])
        p = tl.where(valid_scores, p, 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision=pv_precision)
        ds = p * (dp - delta[:, None])
        dq = tl.dot(ds.to(input_dtype), k, dq, input_precision=qk_precision)

    tl.store(dq_ptr + q_offsets, dq, mask=valid_m[:, None])


@triton.jit
def _streaming_triangle_attention_bwd_dkdv_v1(
    q_ptr,
    k_ptr,
    v_ptr,
    bias1_ptr,
    bias2_ptr,
    lse_ptr,
    delta_ptr,
    do_ptr,
    dk_ptr,
    dv_ptr,
    n_size: tl.constexpr,
    s_size: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    qk_precision_mode: tl.constexpr,
    pv_precision_mode: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty
    qk_precision: tl.constexpr = (
        "tf32x3"
        if qk_precision_mode == 2
        else ("tf32" if qk_precision_mode else "ieee")
    )
    pv_precision: tl.constexpr = (
        "tf32x3"
        if pv_precision_mode == 2
        else ("tf32" if pv_precision_mode else "ieee")
    )
    start_n = tl.program_id(0) * block_n
    h = tl.program_id(1)
    bn = tl.program_id(2)
    b = bn // n_size
    n = bn % n_size

    offs_n = start_n + tl.arange(0, block_n)
    offs_d = tl.arange(0, head_dim)
    valid_n = offs_n < s_size
    base = ((b * n_size + n) * s_size * n_heads + h) * head_dim
    kv_offsets = base + offs_n[:, None] * n_heads * head_dim + offs_d[None, :]
    k = tl.load(k_ptr + kv_offsets, mask=valid_n[:, None], other=0.0)
    v = tl.load(v_ptr + kv_offsets, mask=valid_n[:, None], other=0.0)
    dk = tl.zeros((block_n, head_dim), tl.float32)
    dv = tl.zeros((block_n, head_dim), tl.float32)

    for block_index in range(0, tl.cdiv(s_size, block_m)):
        offs_m = block_index * block_m + tl.arange(0, block_m)
        valid_m = offs_m < s_size
        q_offsets = base + offs_m[:, None] * n_heads * head_dim + offs_d[None, :]
        q = tl.load(q_ptr + q_offsets, mask=valid_m[:, None], other=0.0)
        do = tl.load(do_ptr + q_offsets, mask=valid_m[:, None], other=0.0)
        row_offsets = ((b * n_size + n) * s_size + offs_m) * n_heads + h
        lse = tl.load(lse_ptr + row_offsets, mask=valid_m, other=0.0)
        delta = tl.load(delta_ptr + row_offsets, mask=valid_m, other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision=qk_precision)
        bias1_base = (b * n_size + n) * s_size
        bias1 = tl.load(
            bias1_ptr + bias1_base + offs_n[None, :],
            mask=valid_n[None, :],
            other=-1.0e9,
        )
        bias2_base = (b * n_heads + h) * s_size * s_size
        bias2_offsets = bias2_base + offs_m[:, None] * s_size + offs_n[None, :]
        valid_scores = valid_m[:, None] & valid_n[None, :]
        bias2 = tl.load(bias2_ptr + bias2_offsets, mask=valid_scores, other=0.0)
        scores = (scores + bias1 + bias2) * 1.4426950408889634
        p = tl.exp2(scores - lse[:, None])
        p = tl.where(valid_scores, p, 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision=pv_precision)
        ds = p * (dp - delta[:, None])
        dk = tl.dot(
            tl.trans(ds.to(input_dtype)), q, dk, input_precision=qk_precision
        )
        dv = tl.dot(
            tl.trans(p.to(input_dtype)), do, dv, input_precision=pv_precision
        )

    tl.store(dk_ptr + kv_offsets, dk, mask=valid_n[:, None])
    tl.store(dv_ptr + kv_offsets, dv, mask=valid_n[:, None])


@triton.jit
def _streaming_triangle_attention_bwd_dbias2_v1(
    q_ptr,
    k_ptr,
    v_ptr,
    bias1_ptr,
    bias2_ptr,
    lse_ptr,
    delta_ptr,
    do_ptr,
    dbias2_ptr,
    n_size,
    s_size: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    qk_precision_mode: tl.constexpr,
    pv_precision_mode: tl.constexpr,
):
    qk_precision: tl.constexpr = (
        "tf32x3"
        if qk_precision_mode == 2
        else ("tf32" if qk_precision_mode else "ieee")
    )
    pv_precision: tl.constexpr = (
        "tf32x3"
        if pv_precision_mode == 2
        else ("tf32" if pv_precision_mode else "ieee")
    )
    start_m = tl.program_id(0) * block_m
    start_n = tl.program_id(1) * block_n
    bh = tl.program_id(2)
    b = bh // n_heads
    h = bh % n_heads

    offs_m = start_m + tl.arange(0, block_m)
    offs_n = start_n + tl.arange(0, block_n)
    offs_d = tl.arange(0, head_dim)
    valid_m = offs_m < s_size
    valid_n = offs_n < s_size
    valid_scores = valid_m[:, None] & valid_n[None, :]
    bias2_base = (b * n_heads + h) * s_size * s_size
    bias2_offsets = bias2_base + offs_m[:, None] * s_size + offs_n[None, :]
    bias2 = tl.load(bias2_ptr + bias2_offsets, mask=valid_scores, other=0.0)
    dbias2 = tl.zeros((block_m, block_n), tl.float32)

    for n in range(0, n_size, 1):
        base = ((b * n_size + n) * s_size * n_heads + h) * head_dim
        q_offsets = base + offs_m[:, None] * n_heads * head_dim + offs_d[None, :]
        kv_offsets = base + offs_n[:, None] * n_heads * head_dim + offs_d[None, :]
        q = tl.load(q_ptr + q_offsets, mask=valid_m[:, None], other=0.0)
        k = tl.load(k_ptr + kv_offsets, mask=valid_n[:, None], other=0.0)
        v = tl.load(v_ptr + kv_offsets, mask=valid_n[:, None], other=0.0)
        do = tl.load(do_ptr + q_offsets, mask=valid_m[:, None], other=0.0)
        row_offsets = ((b * n_size + n) * s_size + offs_m) * n_heads + h
        lse = tl.load(lse_ptr + row_offsets, mask=valid_m, other=0.0)
        delta = tl.load(delta_ptr + row_offsets, mask=valid_m, other=0.0)
        bias1_base = (b * n_size + n) * s_size
        bias1 = tl.load(
            bias1_ptr + bias1_base + offs_n[None, :],
            mask=valid_n[None, :],
            other=-1.0e9,
        )
        scores = tl.dot(q, tl.trans(k), input_precision=qk_precision)
        scores = (scores + bias1 + bias2) * 1.4426950408889634
        p = tl.exp2(scores - lse[:, None])
        p = tl.where(valid_scores, p, 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision=pv_precision)
        dbias2 += p * (dp - delta[:, None])

    tl.store(dbias2_ptr + bias2_offsets, dbias2, mask=valid_scores)


def _validate_precision(q: torch.Tensor, precision: str) -> str:
    try:
        internal = _PRECISION_TO_INTERNAL[precision]
    except KeyError as exc:
        choices = ", ".join(_PRECISION_TO_INTERNAL)
        raise ValueError(f"precision must be one of {choices}; got {precision!r}") from exc
    if q.dtype != torch.float32 and precision != "ieee":
        raise ValueError(f"precision={precision!r} is only valid for FP32 inputs")
    return internal


def _bnshd_triangle_attention_support_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
) -> str | None:
    """Return why the strict streaming implementation cannot accept an input."""
    if q.ndim != 5:
        return f"Q must be rank 5, got rank {q.ndim}"
    if q.shape != k.shape or q.shape != v.shape:
        return "Q, K and V must have identical shapes"
    b, n, s, h, d = q.shape
    if min(b, n, s, h) <= 0:
        return f"B/N/S/H must be positive, got {(b, n, s, h)}"
    if d not in (16, 32, 64):
        return f"head dimension must be one of 16/32/64, got {d}"
    if not q.is_cuda:
        return "Q/K/V must be on a CUDA-compatible FlagOS device"
    supported_dtypes = (torch.float32, torch.bfloat16, torch.float16)
    if q.dtype not in supported_dtypes:
        return (
            "the streaming kernel requires FP32/BF16/FP16, "
            f"got {q.dtype}"
        )
    tensors = {"K": k, "V": v, "Bias1": bias1, "Bias2": bias2}
    for name, tensor in tensors.items():
        allowed_dtypes = (q.dtype, torch.float32) if name.startswith("Bias") else (q.dtype,)
        if tensor.dtype not in allowed_dtypes:
            return (
                f"{name} dtype {tensor.dtype} must be Q dtype {q.dtype}"
                + (" or FP32" if name.startswith("Bias") else "")
            )
        if tensor.device != q.device:
            return f"{name} device {tensor.device} does not match Q device {q.device}"
    expected_bias1 = (b, n, 1, 1, s)
    expected_bias2 = (b, 1, h, s, s)
    if bias1.shape != expected_bias1:
        return f"Bias1 shape must be {expected_bias1}, got {tuple(bias1.shape)}"
    if bias2.shape != expected_bias2:
        return f"Bias2 shape must be {expected_bias2}, got {tuple(bias2.shape)}"
    return None


def _triangle_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
    precision: str,
) -> torch.Tensor:
    """Run the validated streaming inference kernel."""
    out, _ = _streaming_triangle_attention_forward_impl(
        q, k, v, bias1, bias2, precision=precision, save_lse=False
    )
    return out


def _triangle_attention_forward_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run training forward and save one base-2 LSE value per query row."""
    out, lse = _streaming_triangle_attention_forward_impl(
        q, k, v, bias1, bias2, precision=precision, save_lse=True
    )
    assert lse is not None
    return out, lse


def _streaming_triangle_attention_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
    *,
    precision: str,
    save_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    support_error = _bnshd_triangle_attention_support_error(q, k, v, bias1, bias2)
    if support_error is not None:
        raise ValueError(support_error)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    bias1 = bias1.contiguous()
    bias2 = bias2.contiguous()
    b, n, s, h, d = q.shape
    out = torch.empty_like(q)
    lse = (
        torch.empty((b, n, s, h), device=q.device, dtype=torch.float32)
        if save_lse
        else None
    )
    # Start from the conservative cross-device configuration and apply only
    # device/shape configurations measured by the release benchmark.
    block_m, block_n, num_warps, num_stages = 32, 32, 1, 1
    tf32_mode = _validate_precision(q, precision)
    device_name = torch.cuda.get_device_name(q.device)
    h100_device = "NVIDIA H100" in device_name and d == 32 and s >= 128
    bias_seeded_qk = h100_device
    ppu_full_fast = (
        "PPU-ZW810E" in device_name
        and d == 32
        and s >= 128
        and tf32_mode == "full"
    )
    if h100_device and tf32_mode == "pv":
        block_m, block_n, num_warps, num_stages = 128, 32, 4, 1
    elif h100_device and tf32_mode != "none":
        block_m, block_n, num_warps, num_stages = 64, 32, 4, 1
    elif h100_device:
        if h == 2:
            block_m, block_n, num_warps, num_stages = 64, 32, 4, 2
        else:
            block_m, block_n, num_warps, num_stages = 32, 64, 2, 2
    elif ppu_full_fast:
        block_m, block_n, num_warps, num_stages = 64, 64, 4, 1
    _streaming_triangle_attention_fwd_ieee_v1[
        (triton.cdiv(s, block_m), h, b * n)
    ](
        q,
        k,
        v,
        bias1,
        bias2,
        out,
        lse if lse is not None else out,
        n_size=n,
        s_size=s,
        n_heads=h,
        head_dim=d,
        block_m=block_m,
        block_n=block_n,
        save_lse=save_lse,
        bias_seeded_qk=bias_seeded_qk,
        qk_allow_tf32=(
            2
            if tf32_mode in {"x3", "qkx3"}
            else int(tf32_mode in {"full", "qk", "pvx3"})
        ),
        pv_allow_tf32=(
            2
            if tf32_mode in {"x3", "pvx3"}
            else int(tf32_mode in {"full", "pv", "qkx3"})
        ),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    global _PHYSICAL_FORWARD_CALLS
    _PHYSICAL_FORWARD_CALLS += 1
    _FORWARD_PRECISION_COUNTS[tf32_mode] = (
        _FORWARD_PRECISION_COUNTS.get(tf32_mode, 0) + 1
    )
    phase = "training_forward" if save_lse else "inference_forward"
    key = phase + ":" + "x".join(str(value) for value in q.shape)
    _SHAPE_COUNTS[key] = _SHAPE_COUNTS.get(key, 0) + 1
    return out, lse if save_lse else None


def _triangle_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute attention tiles and return dQ/dK/dV/dBias2."""
    support_error = _bnshd_triangle_attention_support_error(q, k, v, bias1, bias2)
    if support_error is not None:
        raise ValueError(support_error)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    bias1 = bias1.contiguous()
    bias2 = bias2.contiguous()
    out = out.contiguous()
    lse = lse.contiguous()
    do = do.contiguous()
    b, n, s, h, d = q.shape
    expected_lse_shape = (b, n, s, h)
    if lse.shape != expected_lse_shape or lse.dtype != torch.float32:
        raise ValueError(f"Expected FP32 LSE with shape {expected_lse_shape}")
    for name, tensor in {"output": out, "output gradient": do}.items():
        if tensor.shape != q.shape:
            raise ValueError(
                f"Expected {name} shape {tuple(q.shape)}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != q.dtype:
            raise ValueError(
                f"Expected {name} dtype {q.dtype}, got {tensor.dtype}"
            )
        if tensor.device != q.device:
            raise ValueError(
                f"Expected {name} device {q.device}, got {tensor.device}"
            )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbias2 = torch.empty_like(bias2)
    delta = torch.empty_like(lse)
    block_m, block_n = 32, 32
    use_large_s_fp32_d32 = q.dtype == torch.float32 and d == 32 and s >= 128
    tf32_mode = _validate_precision(q, precision)
    # Match backward precision to the forward decomposition. QK controls score
    # reconstruction plus dQ/dK; PV controls dP plus dV. Value 2 requests the
    # higher-accuracy TF32x3 dot variant supported by the forward kernel.
    qk_precision_mode = (
        2
        if tf32_mode in {"x3", "qkx3"}
        else int(tf32_mode in {"full", "qk", "pvx3"})
    )
    pv_precision_mode = (
        2
        if tf32_mode in {"x3", "pvx3"}
        else int(tf32_mode in {"full", "pv", "qkx3"})
    )
    if use_large_s_fp32_d32:
        dq_block_m, dq_block_n, dq_warps = 64, 32, 2
        dkdv_block_m, dkdv_block_n, dkdv_warps = 64, 32, 2
        dbias_block_m, dbias_block_n, dbias_warps = 32, 64, 2
        config_name = "large_s_fp32_d32_v2"
    else:
        dq_block_m, dq_block_n, dq_warps = 32, 32, 1
        dkdv_block_m, dkdv_block_n, dkdv_warps = 32, 32, 1
        dbias_block_m, dbias_block_n, dbias_warps = 32, 32, 1
        config_name = "generic_v1"

    row_grid = (triton.cdiv(s, block_m), h, b * n)
    _streaming_triangle_attention_bwd_preprocess_v1[row_grid](
        out,
        do,
        delta,
        n_size=n,
        s_size=s,
        n_heads=h,
        head_dim=d,
        block_m=block_m,
        num_warps=1,
        num_stages=1,
    )
    dq_grid = (triton.cdiv(s, dq_block_m), h, b * n)
    _streaming_triangle_attention_bwd_dq_v1[dq_grid](
        q,
        k,
        v,
        bias1,
        bias2,
        lse,
        delta,
        do,
        dq,
        n_size=n,
        s_size=s,
        n_heads=h,
        head_dim=d,
        block_m=dq_block_m,
        block_n=dq_block_n,
        qk_precision_mode=qk_precision_mode,
        pv_precision_mode=pv_precision_mode,
        num_warps=dq_warps,
        num_stages=1,
    )
    key_grid = (triton.cdiv(s, dkdv_block_n), h, b * n)
    _streaming_triangle_attention_bwd_dkdv_v1[key_grid](
        q,
        k,
        v,
        bias1,
        bias2,
        lse,
        delta,
        do,
        dk,
        dv,
        n_size=n,
        s_size=s,
        n_heads=h,
        head_dim=d,
        block_m=dkdv_block_m,
        block_n=dkdv_block_n,
        qk_precision_mode=qk_precision_mode,
        pv_precision_mode=pv_precision_mode,
        num_warps=dkdv_warps,
        num_stages=1,
    )
    bias_grid = (
        triton.cdiv(s, dbias_block_m),
        triton.cdiv(s, dbias_block_n),
        b * h,
    )
    _streaming_triangle_attention_bwd_dbias2_v1[bias_grid](
        q,
        k,
        v,
        bias1,
        bias2,
        lse,
        delta,
        do,
        dbias2,
        n_size=n,
        s_size=s,
        n_heads=h,
        head_dim=d,
        block_m=dbias_block_m,
        block_n=dbias_block_n,
        qk_precision_mode=qk_precision_mode,
        pv_precision_mode=pv_precision_mode,
        num_warps=dbias_warps,
        num_stages=1,
    )

    global _PHYSICAL_BACKWARD_CALLS
    _PHYSICAL_BACKWARD_CALLS += 1
    key = "backward:" + "x".join(str(value) for value in q.shape)
    _SHAPE_COUNTS[key] = _SHAPE_COUNTS.get(key, 0) + 1
    config_key = f"{config_name}:precision_{tf32_mode}"
    _BACKWARD_CONFIG_COUNTS[config_key] = (
        _BACKWARD_CONFIG_COUNTS.get(config_key, 0) + 1
    )
    return dq, dk, dv, dbias2


def _normalize_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be BNSHD or BNHSD; got {layout!r}")
    if layout == "BNSHD":
        return q, k, v, False
    return (
        q.permute(0, 1, 3, 2, 4),
        k.permute(0, 1, 3, 2, 4),
        v.permute(0, 1, 3, 2, 4),
        True,
    )


def triangle_attention_support_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
    *,
    precision: str = "ieee",
    layout: str = "BNSHD",
) -> str | None:
    """Return ``None`` when an input is supported, otherwise a reason."""
    try:
        q, k, v, _ = _normalize_layout(q, k, v, layout)
        _validate_precision(q, precision)
    except (TypeError, ValueError) as exc:
        return str(exc)
    return _bnshd_triangle_attention_support_error(q, k, v, bias1, bias2)


class _TriangleAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias1: torch.Tensor,
        bias2: torch.Tensor,
        precision: str,
    ) -> torch.Tensor:
        if ctx.needs_input_grad[3]:
            raise RuntimeError(
                "Triangle Attention bias1 is an additive mask and must not "
                "require gradients"
            )
        error = _bnshd_triangle_attention_support_error(q, k, v, bias1, bias2)
        if error is not None:
            raise ValueError(error)
        _validate_precision(q, precision)
        needs_backward = any(ctx.needs_input_grad[i] for i in (0, 1, 2, 4))
        ctx.precision = precision
        if needs_backward:
            out, lse = _triangle_attention_forward_with_lse(
                q, k, v, bias1, bias2, precision
            )
            ctx.save_for_backward(q, k, v, bias1, bias2, out, lse)
            return out
        return _triangle_attention_forward(q, k, v, bias1, bias2, precision)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, bias1, bias2, out, lse = ctx.saved_tensors
        dq, dk, dv, dbias2 = _triangle_attention_backward(
            q,
            k,
            v,
            bias1,
            bias2,
            out,
            lse,
            grad_output,
            ctx.precision,
        )
        return dq, dk, dv, None, dbias2, None


def triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
    *,
    precision: str = "ieee",
    layout: str = "BNSHD",
) -> torch.Tensor:
    """Compute streaming Triangle Attention without materializing S-by-S scores.

    ``BNSHD`` inputs have shape ``(batch, sequence_group, sequence, heads,
    head_dim)``. ``BNHSD`` inputs swap the sequence and heads axes. ``bias1``
    is a non-learnable additive mask with shape ``(B, N, 1, 1, S)`` and
    ``bias2`` has shape ``(B, 1, H, S, S)``. First-order gradients are
    implemented for Q, K, V and bias2; unsupported inputs fail without fallback.
    """
    q, k, v, transpose_output = _normalize_layout(q, k, v, layout)
    error = triangle_attention_support_error(
        q, k, v, bias1, bias2, precision=precision, layout="BNSHD"
    )
    if error is not None:
        raise ValueError(error)
    out = _TriangleAttentionFunction.apply(q, k, v, bias1, bias2, precision)
    return out.permute(0, 1, 3, 2, 4) if transpose_output else out


class TriangleAttention(torch.nn.Module):
    """Module wrapper for :func:`triangle_attention`."""

    def __init__(self, *, precision: str = "ieee", layout: str = "BNSHD") -> None:
        super().__init__()
        if precision not in _PRECISION_TO_INTERNAL:
            raise ValueError(f"unsupported precision {precision!r}")
        if layout not in _LAYOUTS:
            raise ValueError(f"unsupported layout {layout!r}")
        self.precision = precision
        self.layout = layout

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias1: torch.Tensor,
        bias2: torch.Tensor,
    ) -> torch.Tensor:
        return triangle_attention(
            q,
            k,
            v,
            bias1,
            bias2,
            precision=self.precision,
            layout=self.layout,
        )
