import logging

import torch
import flag_gems
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _cdist_p_kernel(x1, x2, out, p, P: tl.constexpr, R: tl.constexpr,
                    M: tl.constexpr, BLOCK_M: tl.constexpr):
    pair = tl.program_id(0)
    batch = tl.program_id(1)
    i = pair // R
    j = pair - i * R
    cols = tl.arange(0, BLOCK_M)
    mask = cols < M
    a = tl.load(x1 + batch * P * M + i * M + cols, mask=mask, other=0.0)
    b = tl.load(x2 + batch * R * M + j * M + cols, mask=mask, other=0.0)
    diff = a.to(tl.float32) - b.to(tl.float32)
    abs_diff = tl.abs(diff)
    if p == 0.0:
        # PyTorch's zero-norm cdist is the Hamming count.
        dist = tl.sum(tl.where(abs_diff != 0.0, 1.0, 0.0), axis=0)
    elif p == 1.0:
        dist = tl.sum(abs_diff, axis=0)
    elif p == float("inf"):
        dist = tl.max(abs_diff, axis=0)
    elif p == 2.0:
        dist = tl_extra_shim.sqrt_rn(tl.sum(diff * diff, axis=0))
    else:
        powered = tl_extra_shim.pow(abs_diff, p)
        dist = tl_extra_shim.pow(tl.sum(powered, axis=0), 1.0 / p)
    tl.store(out + batch * P * R + pair, dist)


@libentry()
@triton.jit
def _cdist_l2_grad_x1_reduce_kernel(
    x1, x2, dist, grad_out, grad_x1,
    P: tl.constexpr, R: tl.constexpr, M: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """L2 backward with native CUDA's four-accumulator reduction order.

    Native CUDA ``sum`` keeps four independent accumulators per thread, loads
    values in groups of four, and combines the accumulators afterwards.
    A backend ``tl.sum`` may choose a different tree and change the final
    FP32 bit, so this order is spelled out explicitly.
    """
    idx = tl.program_id(0)
    k = idx % M
    point = idx // M
    i = point % P
    batch = point // P
    a = tl.load(x1 + idx).to(tl.float32)
    acc0 = tl.zeros((), dtype=tl.float32)
    acc1 = tl.zeros((), dtype=tl.float32)
    acc2 = tl.zeros((), dtype=tl.float32)
    acc3 = tl.zeros((), dtype=tl.float32)
    for start in range(0, R, 4):
        for lane in range(4):
            j = start + lane
            valid = j < R
            b = tl.load(x2 + batch * R * M + j * M + k, mask=valid, other=0.0).to(tl.float32)
            dij = tl.load(dist + batch * P * R + i * R + j, mask=valid, other=1.0).to(tl.float32)
            go = tl.load(grad_out + batch * P * R + i * R + j, mask=valid, other=0.0).to(tl.float32)
            term = tl_extra_shim.div_rn(tl_extra_shim.mul_rn(go, a - b), dij)
            term = tl.where(valid & (dij > 0.0), term, 0.0)
            if lane == 0:
                acc0 = tl_extra_shim.add_rn(acc0, term)
            elif lane == 1:
                acc1 = tl_extra_shim.add_rn(acc1, term)
            elif lane == 2:
                acc2 = tl_extra_shim.add_rn(acc2, term)
            else:
                acc3 = tl_extra_shim.add_rn(acc3, term)
    acc = tl_extra_shim.add_rn(acc0, acc1)
    acc = tl_extra_shim.add_rn(acc, acc2)
    acc = tl_extra_shim.add_rn(acc, acc3)
    tl.store(grad_x1 + idx, acc)


@libentry()
@triton.jit
def _cdist_l2_grad_x2_reduce_kernel(
    x1, x2, dist, grad_out, grad_x2,
    P: tl.constexpr, R: tl.constexpr, M: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """L2 backward for x2 using native CUDA's four-accumulator order."""
    idx = tl.program_id(0)
    k = idx % M
    point = idx // M
    j = point % R
    batch = point // R
    b = tl.load(x2 + idx).to(tl.float32)
    acc0 = tl.zeros((), dtype=tl.float32)
    acc1 = tl.zeros((), dtype=tl.float32)
    acc2 = tl.zeros((), dtype=tl.float32)
    acc3 = tl.zeros((), dtype=tl.float32)
    for start in range(0, P, 4):
        for lane in range(4):
            i = start + lane
            valid = i < P
            a = tl.load(x1 + batch * P * M + i * M + k, mask=valid, other=0.0).to(tl.float32)
            dij = tl.load(dist + batch * P * R + i * R + j, mask=valid, other=1.0).to(tl.float32)
            go = tl.load(grad_out + batch * P * R + i * R + j, mask=valid, other=0.0).to(tl.float32)
            term = tl_extra_shim.div_rn(tl_extra_shim.mul_rn(go, b - a), dij)
            term = tl.where(valid & (dij > 0.0), term, 0.0)
            if lane == 0:
                acc0 = tl_extra_shim.add_rn(acc0, term)
            elif lane == 1:
                acc1 = tl_extra_shim.add_rn(acc1, term)
            elif lane == 2:
                acc2 = tl_extra_shim.add_rn(acc2, term)
            else:
                acc3 = tl_extra_shim.add_rn(acc3, term)
    acc = tl_extra_shim.add_rn(acc0, acc1)
    acc = tl_extra_shim.add_rn(acc, acc2)
    acc = tl_extra_shim.add_rn(acc, acc3)
    tl.store(grad_x2 + idx, acc)


@libentry()
@triton.jit
def _cdist_p_grad_x1_kernel(x1, x2, dist, grad_out, grad_x1,
                            p, P: tl.constexpr, R: tl.constexpr, M: tl.constexpr):
    """Generic-p x1 gradient with CUDA Reduce.cuh's four streams."""
    idx = tl.program_id(0)
    k = idx % M
    point = idx // M
    i = point % P
    batch = point // P
    a = tl.load(x1 + idx).to(tl.float32)
    acc0 = tl.zeros((), dtype=tl.float32)
    acc1 = tl.zeros((), dtype=tl.float32)
    acc2 = tl.zeros((), dtype=tl.float32)
    acc3 = tl.zeros((), dtype=tl.float32)
    for start in range(0, R, 4):
        for lane in range(4):
            j = start + lane
            valid = j < R
            dij = tl.load(dist + batch * P * R + i * R + j, mask=valid, other=1.0).to(tl.float32)
            go = tl.load(grad_out + batch * P * R + i * R + j, mask=valid, other=0.0).to(tl.float32)
            b = tl.load(x2 + batch * R * M + j * M + k, mask=valid, other=0.0).to(tl.float32)
            diff = tl_extra_shim.sub_rn(a, b)
            abs_diff = tl.abs(diff)
            if p == 0.0:
                local = tl.zeros_like(diff)
            elif p == 1.0:
                local = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
            elif p == float("inf"):
                sign = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
                local = tl.where((dij > 0.0) & (abs_diff == dij), sign, 0.0)
            elif p == 2.0:
                local = tl_extra_shim.div_rn(diff, dij)
                local = tl.where(dij > 0.0, local, 0.0)
            else:
                numerator = tl_extra_shim.pow(abs_diff, p - 1.0)
                denominator = tl_extra_shim.pow(dij, p - 1.0)
                sign = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
                local = tl_extra_shim.div_rn(numerator, denominator)
                local = tl.where((dij > 0.0) & (abs_diff > 0.0), sign * local, 0.0)
            term = tl_extra_shim.mul_rn(go, local)
            term = tl.where(valid, term, 0.0)
            if lane == 0:
                acc0 = tl_extra_shim.add_rn(acc0, term)
            elif lane == 1:
                acc1 = tl_extra_shim.add_rn(acc1, term)
            elif lane == 2:
                acc2 = tl_extra_shim.add_rn(acc2, term)
            else:
                acc3 = tl_extra_shim.add_rn(acc3, term)
    acc = tl_extra_shim.add_rn(acc0, acc1)
    acc = tl_extra_shim.add_rn(acc, acc2)
    acc = tl_extra_shim.add_rn(acc, acc3)
    tl.store(grad_x1 + idx, acc)

@libentry()
@triton.jit
def _cdist_p_grad_x2_kernel(x1, x2, dist, grad_out, grad_x2,
                            p, P: tl.constexpr, R: tl.constexpr, M: tl.constexpr):
    """Generic-p x2 gradient with CUDA Reduce.cuh's four streams."""
    idx = tl.program_id(0)
    k = idx % M
    point = idx // M
    j = point % R
    batch = point // R
    b = tl.load(x2 + idx).to(tl.float32)
    acc0 = tl.zeros((), dtype=tl.float32)
    acc1 = tl.zeros((), dtype=tl.float32)
    acc2 = tl.zeros((), dtype=tl.float32)
    acc3 = tl.zeros((), dtype=tl.float32)
    for start in range(0, P, 4):
        for lane in range(4):
            i = start + lane
            valid = i < P
            dij = tl.load(dist + batch * P * R + i * R + j, mask=valid, other=1.0).to(tl.float32)
            go = tl.load(grad_out + batch * P * R + i * R + j, mask=valid, other=0.0).to(tl.float32)
            a = tl.load(x1 + batch * P * M + i * M + k, mask=valid, other=0.0).to(tl.float32)
            diff = tl_extra_shim.sub_rn(b, a)
            abs_diff = tl.abs(diff)
            if p == 0.0:
                local = tl.zeros_like(diff)
            elif p == 1.0:
                local = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
            elif p == float("inf"):
                sign = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
                local = tl.where((dij > 0.0) & (abs_diff == dij), sign, 0.0)
            elif p == 2.0:
                local = tl_extra_shim.div_rn(diff, dij)
                local = tl.where(dij > 0.0, local, 0.0)
            else:
                numerator = tl_extra_shim.pow(abs_diff, p - 1.0)
                denominator = tl_extra_shim.pow(dij, p - 1.0)
                sign = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
                local = tl_extra_shim.div_rn(numerator, denominator)
                local = tl.where((dij > 0.0) & (abs_diff > 0.0), sign * local, 0.0)
            term = tl_extra_shim.mul_rn(go, local)
            term = tl.where(valid, term, 0.0)
            if lane == 0:
                acc0 = tl_extra_shim.add_rn(acc0, term)
            elif lane == 1:
                acc1 = tl_extra_shim.add_rn(acc1, term)
            elif lane == 2:
                acc2 = tl_extra_shim.add_rn(acc2, term)
            else:
                acc3 = tl_extra_shim.add_rn(acc3, term)
    acc = tl_extra_shim.add_rn(acc0, acc1)
    acc = tl_extra_shim.add_rn(acc, acc2)
    acc = tl_extra_shim.add_rn(acc, acc3)
    tl.store(grad_x2 + idx, acc)

def _cdist_forward(x1, x2, p):
    batch_shape = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
    points, features = x1.shape[-2:]
    refs = x2.shape[-2]
    x1c = x1.expand(*batch_shape, points, features).contiguous()
    x2c = x2.expand(*batch_shape, refs, features).contiguous()
    out = torch.empty((*batch_shape, points, refs), device=x1.device, dtype=x1.dtype)
    if features == 0:
        out = torch.zeros((*batch_shape, points, refs), device=x1.device, dtype=x1.dtype)
    elif points and refs and out.numel():
        batches = out.numel() // (points * refs)
        _cdist_p_kernel[(points * refs, batches)](
            x1c, x2c, out, float(p), P=points, R=refs, M=features,
            BLOCK_M=triton.next_power_of_2(features), num_warps=1, num_stages=1)
    return x1c, x2c, out


def _cdist_setup_context(ctx, inputs, output):
    x1, x2, p, _compute_mode = inputs
    ctx.p = float(p)
    ctx.save_for_backward(x1, x2, output)


def _cdist_backward(ctx, grad_out):
    x1, x2, dist = ctx.saved_tensors
    p = ctx.p
    batch_shape = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
    points, features = x1.shape[-2:]
    refs = x2.shape[-2]
    x1c = x1.expand(*batch_shape, points, features).contiguous()
    x2c = x2.expand(*batch_shape, refs, features).contiguous()
    if grad_out.numel() == 0 or features == 0:
        return (
            torch.zeros_like(x1) if ctx.needs_input_grad[0] else None,
            torch.zeros_like(x2) if ctx.needs_input_grad[1] else None,
            None,
            None,
        )
    grad_out = grad_out.contiguous()
    gx1 = gx2 = None
    with torch_device_fn.device(x1.device):
        if ctx.needs_input_grad[0]:
            gx1 = torch.empty_like(x1c)
            if gx1.numel():
                if p == 2.0:
                    block_r = min(triton.next_power_of_2(refs), 64)
                    _cdist_l2_grad_x1_reduce_kernel[(gx1.numel(),)](
                        x1c, x2c, dist, grad_out, gx1,
                        P=points, R=refs, M=features, BLOCK_R=block_r,
                        num_warps=1, num_stages=1)
                else:
                    _cdist_p_grad_x1_kernel[(gx1.numel(),)](
                        x1c, x2c, dist, grad_out, gx1, float(p),
                        P=points, R=refs, M=features, num_warps=1, num_stages=1)
            gx1 = flag_gems.sum_to_size(gx1, x1.shape)
        if ctx.needs_input_grad[1]:
            gx2 = torch.empty_like(x2c)
            if gx2.numel():
                if p == 2.0:
                    block_p = min(triton.next_power_of_2(points), 64)
                    _cdist_l2_grad_x2_reduce_kernel[(gx2.numel(),)](
                        x1c, x2c, dist, grad_out, gx2,
                        P=points, R=refs, M=features, BLOCK_P=block_p,
                        num_warps=1, num_stages=1)
                else:
                    _cdist_p_grad_x2_kernel[(gx2.numel(),)](
                        x1c, x2c, dist, grad_out, gx2, float(p),
                        P=points, R=refs, M=features, num_warps=1, num_stages=1)
            gx2 = flag_gems.sum_to_size(gx2, x2.shape)
    return gx1, gx2, None, None


class _CDistFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, p, compute_mode):
        output = _cdist_forward(x1, x2, p)[2]
        _cdist_setup_context(ctx, (x1, x2, p, compute_mode), output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return _cdist_backward(ctx, grad_output)



def cdist(x1, x2, p=2.0, compute_mode=None):
    logger.debug("GEMS CDIST")
    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError("cdist expects tensors with at least two dimensions")
    if x1.shape[-1] != x2.shape[-1]:
        raise ValueError("cdist feature dimensions must match")
    if x1.device != x2.device or x1.dtype != x2.dtype:
        raise ValueError("cdist inputs must share device and dtype")
    if float(p) < 0.0:
        raise NotImplementedError("FlagGems PPU cdist supports p >= 0 only")
    if x1.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(f"FlagGems PPU cdist does not support {x1.dtype}")
    with torch_device_fn.device(x1.device):
        return _CDistFunction.apply(x1, x2, float(p), compute_mode)
