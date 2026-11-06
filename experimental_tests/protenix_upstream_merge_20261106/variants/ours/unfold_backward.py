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

"""Output-owned unfold backward: each element sums overlapping windows in order."""
import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _unfold_backward_gather(grad, output, total, after, length, dimension,
                            SIZE: tl.constexpr, STEP: tl.constexpr,
                            BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < total
    column = index % after
    position = (index // after) % dimension
    batch = index // (after * dimension)
    first = tl.maximum(0, (position - SIZE + STEP) // STEP)
    last = tl.minimum(length - 1, position // STEP)
    if grad.dtype.element_ty == tl.float64:
        acc = tl.full((BLOCK,), 0, tl.float64)
    else:
        acc = tl.full((BLOCK,), 0, tl.float32)
    for offset in range(triton.cdiv(SIZE, STEP)):
        window = first + offset
        within = position - window * STEP
        source = ((batch * length + window) * after + column) * SIZE + within
        value = tl.load(grad + source, mask=valid & (window <= last), other=0)
        acc = acc + value.to(acc.dtype)
    tl.store(output + index, acc, mask=valid)


def unfold_backward(grad_in: torch.Tensor, input_sizes, dim: int,
                    size: int, step: int) -> torch.Tensor:
    logger.debug('GEMS DETERMINISTIC UNFOLD BACKWARD')
    input_sizes = tuple(int(x) for x in input_sizes)
    if not input_sizes or not -len(input_sizes) <= dim < len(input_sizes):
        raise IndexError('unfold_backward dimension out of range')
    dim %= len(input_sizes)
    if step <= 0 or not 0 <= size <= input_sizes[dim]:
        raise ValueError('invalid unfold size or step')
    if grad_in.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise NotImplementedError('unfold_backward candidate supports floating-point gradients')
    length = (input_sizes[dim] - size) // step + 1
    expected = (*input_sizes[:dim], length, *input_sizes[dim + 1:], size)
    if tuple(grad_in.shape) != expected:
        raise ValueError(f'unfold gradient shape {tuple(grad_in.shape)} != {expected}')
    result = torch.empty(input_sizes, device=grad_in.device, dtype=grad_in.dtype)
    if not result.numel():
        return result
    if size == 0:
        return torch.zeros_like(result)
    gradient = grad_in.contiguous()
    after = math.prod(input_sizes[dim + 1:])
    _unfold_backward_gather[(triton.cdiv(result.numel(), 256),)](
        gradient, result, result.numel(), after, length, input_sizes[dim],
        SIZE=size, STEP=step, BLOCK=256, num_warps=4)
    return result
