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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.unfold_backward
@pytest.mark.parametrize(
    "input_sizes, dim, size, step",
    [
        ((32, 64), 1, 16, 16),
        ((16, 33), 0, 5, 2),
        ((4, 8, 12), -1, 6, 4),
        ((7, 13), 1, 13, 3),
        ((6, 20), 1, 7, 4),
        ((2, 3, 17), -1, 9, 1),
        ((2, 17), 1, 4, 6),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_unfold_backward(input_sizes, dim, size, step, dtype):
    d = dim % len(input_sizes)
    num_windows = (input_sizes[d] - size) // step + 1
    grad_shape = (
        list(input_sizes[:d]) + [num_windows] + list(input_sizes[d + 1 :]) + [size]
    )

    grad_in = torch.randn(grad_shape, dtype=dtype, device=flag_gems.device)

    ref_grad = utils.to_reference(grad_in, True)
    ref_out = torch.ops.aten.unfold_backward(ref_grad, input_sizes, dim, size, step)

    with flag_gems.use_gems():
        res_out = flag_gems.unfold_backward(grad_in, input_sizes, dim, size, step)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=size)


@pytest.mark.unfold_backward
def test_unfold_backward_noncontiguous_grad_input():
    """The backward kernel must honor logical values from a view."""
    input_sizes = (2, 8, 3)
    dim, size, step = 1, 3, 2
    num_windows = (input_sizes[dim] - size) // step + 1
    grad_shape = (input_sizes[0], num_windows, input_sizes[2], size)

    base = torch.arange(
        torch.tensor(grad_shape).prod().item(),
        dtype=torch.float32,
        device=flag_gems.device,
    ).reshape(grad_shape)
    grad_in = base.permute(0, 2, 1, 3)
    assert not grad_in.is_contiguous()

    ref_out = torch.ops.aten.unfold_backward(
        utils.to_reference(grad_in), input_sizes, dim, size, step
    )
    with flag_gems.use_gems():
        res_out = flag_gems.unfold_backward(grad_in, input_sizes, dim, size, step)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.unfold_backward
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("size, step", [(128, 32), (7, 11)])
def test_unfold_backward_default_repeatability(dtype, size, step):
    """Overlaps must be repeatable without enabling global deterministic mode."""
    shape = (4, 193, 7)
    x = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
    y = x.unfold(1, size, step)
    generator = torch.Generator().manual_seed(20260825)
    gradient = torch.randn((*y.shape, 2), generator=generator)[..., 0].to(dtype)
    (reference,) = torch.autograd.grad(y, x, gradient.double())
    gpu_gradient = gradient.to(flag_gems.device)
    backing = torch.empty((*gpu_gradient.shape, 2), device=flag_gems.device, dtype=dtype)
    backing[..., 0] = gpu_gradient
    gpu_gradient = backing[..., 0]
    assert not gpu_gradient.is_contiguous()
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        with flag_gems.use_gems():
            results = [
                flag_gems.unfold_backward(gpu_gradient, shape, 1, size, step)
                for _ in range(16)
            ]
        for result in results[1:]:
            assert torch.equal(result, results[0])
        flag_gems.testing.assert_close(results[0].cpu(), reference, dtype, reduce_dim=size)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
