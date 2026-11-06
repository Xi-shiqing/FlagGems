import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for cdist test: (P, M), (R, M) -> (P, R)
# Note: torch.cdist doesn't support float16 on CUDA
CDIST_SHAPES = [
    ((4, 8), (6, 8)),
    ((8, 16), (8, 16)),
    ((16, 32), (16, 32)),
    ((32, 64), (32, 64)),
    ((1, 8), (1, 8)),
    ((2, 3, 8), (2, 4, 8)),  # batch dimension
    ((5, 3, 4), (3, 5, 2, 4)),  # unequal batch dims broadcast (5,) vs (3,5) -> (3,5)
]


@pytest.mark.cdist
@pytest.mark.parametrize("shapes", CDIST_SHAPES)
# torch.cdist doesn't support float16 on CUDA; only float32 is numerically stable
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cdist(shapes, dtype):
    shape1, shape2 = shapes
    x1 = torch.randn(*shape1, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(*shape2, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(x1)
    ref_x2 = utils.to_reference(x2)

    ref_out = torch.cdist(ref_x1, ref_x2, p=2.0)
    res_out = flag_gems.cdist(x1, x2, p=2.0)

    # cdist L2 computation accumulates error over feature dimension (M up to 128);
    # atol=0.01 is sufficient for float32 comparison
    utils.gems_assert_close(res_out, ref_out, dtype, atol=0.01)


@pytest.mark.cdist
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="PPU cdist autograd provider")
@pytest.mark.parametrize("case", ["self", "independent", "broadcast", "strided", "empty_refs", "empty_points", "empty_features"])
def test_cdist_registered_backward(case):
    generator = torch.Generator().manual_seed(20261106)
    shape1, shape2 = (2, 5, 3), (2, 7, 3)
    if case == "broadcast":
        shape1 = (1, 5, 3)
    elif case == "empty_refs":
        shape2 = (2, 0, 3)
    elif case == "empty_points":
        shape1 = (2, 0, 3)
    elif case == "empty_features":
        shape1, shape2 = (2, 5, 0), (2, 7, 0)
    values = [torch.randn(shape1, generator=generator)]
    if case != "self":
        values.append(torch.randn(shape2, generator=generator))
    if case == "strided":
        values = [value.transpose(0, 1).contiguous().transpose(0, 1) for value in values]
    reference_inputs = [value.double().requires_grad_() for value in values]
    def operation(inputs):
        return torch.cdist(inputs[0], inputs[0] if case == "self" else inputs[1],
                           p=2, compute_mode="donot_use_mm_for_euclid_dist")
    reference = operation(reference_inputs)
    upstream = torch.randn(reference.shape, generator=generator)
    expected = torch.autograd.grad(reference, reference_inputs, upstream.double())
    for _ in range(2):
        device_inputs = [value.to(flag_gems.device).requires_grad_() for value in values]
        with flag_gems.use_gems(exclude=[]):
            actual = operation(device_inputs)
            gradients = torch.autograd.grad(actual, device_inputs, upstream.to(flag_gems.device))
            # AutogradCPU stays native while the CUDA context is active.
            cpu_inputs = [value.double().requires_grad_() for value in values]
            cpu_output = operation(cpu_inputs)
            cpu_gradients = torch.autograd.grad(cpu_output, cpu_inputs, upstream.double())
        torch.testing.assert_close(actual.detach().cpu().double(), reference.detach(), rtol=2e-5, atol=2e-5)
        for actual_grad, expected_grad, cpu_grad in zip(gradients, expected, cpu_gradients):
            torch.testing.assert_close(actual_grad.cpu().double(), expected_grad, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(cpu_grad, expected_grad, rtol=0, atol=0)
