# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _unfold_backward_kernel(
    grad_in_ptr,
    grad_out_ptr,
    numel_in,
    prod_after,
    L,
    size,
    step,
    D,
    inner_total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel_in

    vals = tl.load(grad_in_ptr + offs, mask=mask, other=0)
    vals_f32 = tl.cast(vals, tl.float32)

    k = offs % size
    tmp1 = offs // size
    after_lin = tmp1 % prod_after
    tmp2 = offs // (prod_after * size)
    s = tmp2 % L
    before_lin = offs // inner_total

    pos = s * step + k

    out_id = ((before_lin * D) + pos) * prod_after + after_lin

    tl.atomic_add(grad_out_ptr + out_id, vals_f32, mask=mask)


@triton.jit
def _unfold_backward_deterministic_kernel(
    grad_in_ptr,
    grad_out_ptr,
    numel_out,
    prod_after,
    L,
    D,
    SIZE: tl.constexpr,
    STEP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather overlapping windows in a fixed order, without atomics."""
    pid = tl.program_id(0)
    out_id = pid * BLOCK + tl.arange(0, BLOCK)
    out_mask = out_id < numel_out

    before_lin = out_id // (D * prod_after)
    within = out_id % (D * prod_after)
    pos = within // prod_after
    after_lin = within % prod_after

    # Windows containing ``pos`` have s*STEP <= pos < s*STEP+SIZE.
    s_first = tl.maximum(0, (pos - SIZE + STEP) // STEP)
    s_last = tl.minimum(L - 1, pos // STEP)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(0, (SIZE + STEP - 1) // STEP + 1):
        s = s_first + i
        k = pos - s * STEP
        valid = out_mask & (s <= s_last) & (k >= 0) & (k < SIZE)
        in_id = (((before_lin * L + s) * prod_after + after_lin) * SIZE) + k
        value = tl.load(grad_in_ptr + in_id, mask=valid, other=0.0)
        acc += value.to(tl.float32)

    tl.store(grad_out_ptr + out_id, acc, mask=out_mask)


def unfold_backward(
    grad_in: torch.Tensor, input_sizes, dim: int, size: int, step: int
) -> torch.Tensor:
    logger.debug("GEMS UNFOLD BACKWARD")
    if step <= 0:
        raise ValueError("step must be > 0")

    if not isinstance(input_sizes, (list, tuple)):
        input_sizes = list(input_sizes)
    input_sizes = [int(s) for s in input_sizes]
    ndim = len(input_sizes)
    d = dim % ndim

    D = int(input_sizes[d])
    L = (D - int(size)) // int(step) + 1

    prod_after = 1
    for s_ in input_sizes[d + 1 :]:
        prod_after *= int(s_)
    inner_total = int(L) * int(prod_after) * int(size)

    # Autograd may pass a transpose/slice/view with a non-contiguous backing
    # layout, while both kernels below use logical contiguous indexing.
    grad_in = grad_in.contiguous()

    device = grad_in.device
    grad_out_f32 = torch.zeros(input_sizes, dtype=torch.float32, device=device)

    numel_in = grad_in.numel()

    BLOCK = 128
    if torch.are_deterministic_algorithms_enabled():
        numel_out = grad_out_f32.numel()
        grid = lambda meta: (triton.cdiv(numel_out, meta["BLOCK"]),)
        _unfold_backward_deterministic_kernel[grid](
            grad_in,
            grad_out_f32,
            numel_out,
            prod_after,
            L,
            D,
            SIZE=int(size),
            STEP=int(step),
            BLOCK=BLOCK,
        )
    else:
        grid = lambda meta: (triton.cdiv(numel_in, meta["BLOCK"]),)
        _unfold_backward_kernel[grid](
            grad_in,
            grad_out_f32,
            numel_in,
            prod_after,
            L,
            size,
            step,
            D,
            inner_total,
            BLOCK=BLOCK,
        )

    if grad_in.dtype != torch.float32:
        return grad_out_f32.to(grad_in.dtype)
    return grad_out_f32
