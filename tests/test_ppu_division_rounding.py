import math

import pytest
import torch

import flag_gems
from flag_gems.runtime import device as runtime_device


@pytest.mark.true_divide
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU scalar division")
@pytest.mark.parametrize("scalar", [math.sqrt(32), 3.0, 0.1, -1.7])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice"])
def test_ppu_tensor_scalar_division_matches_native(scalar, layout):
    cpu = torch.randn((31, 107), generator=torch.Generator().manual_seed(8030))
    x = cpu.to(flag_gems.device)
    if layout == "transpose":
        x = x.t()
    elif layout == "slice":
        x = x[:, ::2]
    expected = torch.div(x, scalar)
    with flag_gems.use_gems():
        actual = torch.div(x, scalar)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.true_divide
@pytest.mark.true_divide_out
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU reciprocal division")
@pytest.mark.parametrize("scalar_numerator", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
def test_ppu_division_broadcast_and_backward(scalar_numerator, use_out):
    cpu_b = torch.linspace(0.05, 12, 43).reshape(1, 43)
    cpu_a = torch.linspace(-7, 5, 17).reshape(17, 1)
    a = 1.7 if scalar_numerator else cpu_a.to(flag_gems.device).requires_grad_(not use_out)
    b = cpu_b.to(flag_gems.device).requires_grad_(not use_out)
    ref_a = a if scalar_numerator else cpu_a.double().requires_grad_(not use_out)
    ref_b = cpu_b.double().requires_grad_(not use_out)
    expected = ref_a / ref_b
    with flag_gems.use_gems():
        if use_out:
            storage = torch.empty((*expected.shape, 2), device=flag_gems.device)
            actual = flag_gems.true_divide_out(a, b, storage[..., 0])
        else:
            actual = flag_gems.true_divide(a, b) if scalar_numerator else torch.div(a, b)
    torch.testing.assert_close(actual.cpu(), expected.float(), atol=2e-6, rtol=2e-7)
    if not use_out and not scalar_numerator:
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(a.grad.cpu(), ref_a.grad.float(), atol=3e-5, rtol=5e-7)
        torch.testing.assert_close(b.grad.cpu(), ref_b.grad.float(), atol=3e-4, rtol=5e-7)


@pytest.mark.true_divide
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU scalar division")
@pytest.mark.parametrize("scalar", [0.0, -0.0, float("inf"), -float("inf"), 1e-30, 1e30])
def test_ppu_scalar_division_extremes(scalar):
    x = torch.tensor([-3.0, -0.0, 0.0, 3.0], device=flag_gems.device)
    expected = torch.div(x, scalar)
    with flag_gems.use_gems():
        actual = torch.div(x, scalar)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.true_divide
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU scalar division")
def test_ppu_scalar_division_backward():
    x = torch.linspace(-2, 3, 1031, device=flag_gems.device).requires_grad_(True)
    expected = x / -1.7
    expected.sum().backward()
    grad = x.grad.clone()
    x.grad = None
    with flag_gems.use_gems():
        actual = x / -1.7
        actual.sum().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, grad, atol=0, rtol=0)


@pytest.mark.true_divide_
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU scalar division")
@pytest.mark.parametrize("scalar", [math.sqrt(32), -1.7, 0.1])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice"])
def test_ppu_inplace_scalar_division_forward_and_backward(scalar, layout):
    x = torch.linspace(-2, 3, 31 * 43, device=flag_gems.device).reshape(31, 43)
    if layout == "transpose":
        x = x.t()
    elif layout == "slice":
        x = x[:, ::2]
    x = x.detach().requires_grad_(True)
    expected = x.clone()
    expected.div_(scalar)
    expected.sum().backward()
    grad = x.grad.clone()
    x.grad = None
    with flag_gems.use_gems():
        actual = x.clone()
        address = actual.data_ptr()
        actual.div_(scalar)
        actual.sum().backward()
    assert actual.data_ptr() == address
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, grad, atol=0, rtol=0)
