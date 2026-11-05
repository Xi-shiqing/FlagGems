import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.device_info import get_sm_count
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_persistent"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_persistent_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    # using 1d tile makes code clean
    # Map the program id to the row of X and Y it should compute.
    pid = tle.program_id(0)

    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + pid * N + n_offsets, mask, other=0.0).to(tl.float32)
    m = tl.sum(x) / N
    d = x - m  # deviation
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s)  # sum of square of deviation
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=mask)
    out = (x - m) * rstd * w + b

    tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_persistent"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_persistent_kernel_multiline(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tle.program_id(0)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_offsets < M

    n_offsets = tl.arange(0, TILE_N)[None, :]
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask

    x = tl.load(in_ptr + m_offsets[:, None] * N + n_offsets, mask, other=0.0).to(
        tl.float32
    )
    m = tl.sum(x, axis=1) / N
    d = x - m[:, None]  # deviation
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s, axis=1)  # sum of square of deviation
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + m_offsets, m, mask=m_mask)
    tl.store(out_rstd_ptr + m_offsets, rstd, mask=m_mask)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=n_mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=n_mask)
    out = (x - m[:, None]) * rstd[:, None] * w + b

    tl.store(out_ptr + m_offsets[:, None] * N + n_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_loop"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_loop_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tle.program_id(0)

    # Compute mean
    m = tl.zeros((TILE_N,), dtype=tl.float32)  # mean
    s = tl.zeros((TILE_N,), dtype=tl.float32)  # sum((x - m)^2)
    cnt = tl.zeros((TILE_N,), dtype=tl.int32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        new_m = m + (x - m) / (step + 1)
        new_s = s + (x - new_m) * (x - m)
        cnt += 1
        m = new_m
        s = new_s

    # the last step
    for step in range(num_steps - 1, num_steps, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask).to(tl.float32)
        new_m = tl.where(mask, m + (x - m) / (step + 1), m)
        new_s = tl.where(mask, s + (x - new_m) * (x - m), s)
        cnt += mask.to(tl.int32)
        m = new_m
        s = new_s

    final_m = tl.sum(m * cnt) / N
    var = tl.sum(s + cnt * (m - final_m) * (m - final_m)) / N
    rstd = tl.math.rsqrt(var + eps)
    m = final_m
    # Write mean / rstd
    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    # reverse the order of the second sweep
    # Normalize and apply linear transformation
    prev_multiple = prev_multiple_of(N, TILE_N)
    # the first step, masking is needed
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets, mask=mask)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets, mask=mask)
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets, eviction_policy="evict_first").to(
            tl.float32
        )
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets)
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_backward"),
    key=["M", "N"],
)
@triton.jit
def layer_norm_backward_kernel(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = tle.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = pid < M
    dY += pid * N
    X += pid * N
    dX += pid * N
    Mean += pid
    Rstd += pid

    mean = tl.load(Mean, mask=row_mask).to(tl.float32)
    rstd = tl.load(Rstd, mask=row_mask).to(tl.float32)

    dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask and col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        dx_hat = dy * w
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2, axis=1)[:, None]
    dx_3 = tl.sum(dx_part3, axis=1)[:, None]

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask and col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        dx_hat = dy * w
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)
        tl.store(dX + cols, dx, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_bias_backward"),
    key=["N"],
)
@triton.jit
def weight_bias_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = tle.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = pid < N
    dY += pid[None, :]
    X += pid[None, :]
    accW = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    accB = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        row_mask = rows < M
        mask = row_mask and col_mask[None, :]
        dy = tl.load(dY + rows * N, mask).to(tl.float32)
        x = tl.load(X + rows * N, mask).to(tl.float32)
        mean = tl.load(Mean + rows, mask=rows < M).to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=rows < M).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        accW += dy * x * rstd
        accB += dy
    if dW:
        dw = tl.sum(accW, axis=0)
        tl.store(dW + pid, dw, mask=col_mask)
    if dB:
        db = tl.sum(accB, axis=0)
        tl.store(dB + pid, db, mask=col_mask)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("GEMS LAYERNORM FORWARD")

    N = math.prod(normalized_shape)
    M = input.numel() // N

    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    y = torch.empty_like(input)

    # NOTE: when the input is half-precision(either float16 or bfloat16)
    # these statistical data saved for backward is in single precision
    mean = torch.empty(M, dtype=input.dtype, device=input.device)
    rstd = torch.empty(M, dtype=input.dtype, device=input.device)

    with torch_device_fn.device(input.device):
        if N <= 128:
            TILE_N = triton.next_power_of_2(N)
            TILE_M = triton.cdiv(1024, TILE_N)
            grid = (triton.cdiv(M, TILE_M), 1, 1)
            layer_norm_persistent_kernel_multiline[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                TILE_M,
                TILE_N,
            )
        elif N <= 4096:
            TILE_N = triton.next_power_of_2(N)
            grid = (M, 1, 1)
            layer_norm_persistent_kernel[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
                TILE_N,
            )
        else:
            grid = (M, 1, 1)
            layer_norm_loop_kernel[grid](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                M,
                N,
                eps,
            )
    return y, mean, rstd


def layer_norm_backward(
    grad_out,
    input,
    normalized_shape,
    mean,
    rstd,
    weight=None,
    bias=None,
    output_mask=None,
):
    logger.debug("GEMS LAYERNORM BACKWARD")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    N = math.prod(normalized_shape)
    M = input.numel() // N if N else 0
    if M == 0 or N == 0:
        return (torch.empty_like(input) if output_mask[0] else None,
                torch.zeros_like(weight) if output_mask[1] else None,
                torch.zeros_like(bias) if output_mask[2] else None)

    if output_mask[0]:
        in_grad = torch.empty_like(input)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_ROW_SIZE"]), 1, 1)
        with torch_device_fn.device(input.device):
            layer_norm_backward_kernel[grid](
                grad_out, input, weight, mean, rstd, in_grad, M, N
            )
    else:
        in_grad = None

    if output_mask[1] is False and output_mask[2] is False:
        return in_grad, None, None

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_COL_SIZE"]), 1, 1)
    weight_grad = torch.empty_like(weight) if output_mask[1] else None
    bias_grad = torch.empty_like(bias) if output_mask[2] else None
    with torch_device_fn.device(input.device):
        if _split_parameter_gradients(grad_out, input, mean, rstd, weight_grad, bias_grad, M, N):
            return in_grad, weight_grad, bias_grad
        weight_bias_backward_kernel[grid](
            grad_out, input, mean, rstd, weight_grad, bias_grad, M, N
        )
    return in_grad, weight_grad, bias_grad


@libentry()
@triton.jit
def _parameter_partial_kernel(DY, X, MEAN, RSTD, PW, PB, M, N,
                              PARTS: tl.constexpr, HAS_W: tl.constexpr, HAS_B: tl.constexpr,
                              ROWS: tl.constexpr, COLS: tl.constexpr):
    col = tle.program_id(0)*COLS + tl.arange(0,COLS)
    part = tle.program_id(1)
    length = tl.cdiv(M,PARTS)
    start = part*length
    limit = tl.minimum(start+length,M)
    aw = tl.zeros((ROWS,COLS),tl.float32)
    ab = tl.zeros((ROWS,COLS),tl.float32)
    for offset in range(0,length,ROWS):
        row = start+offset+tl.arange(0,ROWS)
        valid = (row[:,None]<limit)&(col[None,:]<N)
        dy = tl.load(DY+row[:,None]*N+col[None,:],valid,other=0).to(tl.float32)
        if HAS_W:
            x = tl.load(X+row[:,None]*N+col[None,:],valid,other=0).to(tl.float32)
            mean = tl.load(MEAN+row,row<limit,other=0).to(tl.float32)
            rstd = tl.load(RSTD+row,row<limit,other=0).to(tl.float32)
            aw += dy*(x-mean[:,None])*rstd[:,None]
        if HAS_B:
            ab += dy
    if HAS_W:
        tl.store(PW+part*N+col,tl.sum(aw,axis=0),col<N)
    if HAS_B:
        tl.store(PB+part*N+col,tl.sum(ab,axis=0),col<N)


@libentry()
@triton.jit
def _parameter_merge_kernel(PW,PB,DW,DB,N,PARTS:tl.constexpr,
                            HAS_W:tl.constexpr,HAS_B:tl.constexpr,
                            REDUCE:tl.constexpr,COLS:tl.constexpr):
    col = tle.program_id(0)*COLS+tl.arange(0,COLS)
    row = tl.arange(0,REDUCE)
    valid = (row[:,None]<PARTS)&(col[None,:]<N)
    if HAS_W:
        values=tl.load(PW+row[:,None]*N+col[None,:],valid,other=0)
        tl.store(DW+col,tl.sum(values,axis=0),col<N)
    if HAS_B:
        values=tl.load(PB+row[:,None]*N+col[None,:],valid,other=0)
        tl.store(DB+col,tl.sum(values,axis=0),col<N)


def _split_parameter_gradients(dy,x,mean,rstd,dw,db,M,N):
    if M<4096 or x.dtype is not torch.float32 or x.device.type!='cuda':
        return False
    if any(t is not None and t.dtype is not torch.float32 for t in (dw,db)):
        return False
    columns=32
    desired=max(triton.cdiv(2*get_sm_count(),triton.cdiv(N,columns)),triton.cdiv(M,4096))
    maximum=min(256,M//128,(64*1024*1024)//(4*N*(int(dw is not None)+int(db is not None))))
    parts=min(desired,maximum)
    if parts<2:
        return False
    pw=torch.empty((parts,N),dtype=torch.float32,device=x.device) if dw is not None else None
    pb=torch.empty((parts,N),dtype=torch.float32,device=x.device) if db is not None else None
    _parameter_partial_kernel[(triton.cdiv(N,columns),parts)](
        dy,x,mean,rstd,pw,pb,M,N,parts,dw is not None,db is not None,128,columns,
        num_warps=4,num_stages=1)
    _parameter_merge_kernel[(triton.cdiv(N,columns),)](
        pw,pb,dw,db,N,parts,dw is not None,db is not None,triton.next_power_of_2(parts),columns,
        num_warps=4,num_stages=1)
    return True
