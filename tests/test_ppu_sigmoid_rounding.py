import pytest
import torch

import flag_gems
from flag_gems.runtime import device as runtime_device


@pytest.mark.sigmoid
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU sigmoid rounding")
@pytest.mark.parametrize("shape", [(), (0,), (1031,), (17, 31), (4097, 127)])
@pytest.mark.parametrize("tf32", [False, True])
def test_ppu_sigmoid_against_double(shape, tf32):
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        generator = torch.Generator().manual_seed(8025)
        cpu = torch.randn(shape, generator=generator) * 12
        if cpu.numel() >= 9:
            cpu.reshape(-1)[:9] = torch.tensor([-float("inf"), -104, -90, -1, -0.0, 0, 1, 90, float("inf")])
        x = cpu.to(flag_gems.device)
        with torch.no_grad(), flag_gems.use_gems():
            actual = torch.sigmoid(x)
        torch.testing.assert_close(actual.cpu(), torch.sigmoid(cpu.double()).float(), atol=6e-8, rtol=2e-7)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.sigmoid
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU sigmoid rounding")
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("layout", ["transpose", "slice", "contiguous"])
def test_ppu_sigmoid_layout_and_backward(layout, inplace):
    cpu = torch.linspace(-12, 12, 17 * 31).reshape(17, 31)
    x = cpu.to(flag_gems.device)
    if layout == "transpose":
        cpu, x = cpu.t(), x.t()
    elif layout == "slice":
        cpu, x = cpu[:, ::2], x[:, ::2]
    x = x.detach().requires_grad_(True)
    with flag_gems.use_gems():
        y = torch.sigmoid_(x.clone()) if inplace else torch.sigmoid(x)
        y.sum().backward()
    expected = torch.sigmoid(cpu.double())
    torch.testing.assert_close(y.cpu(), expected.float(), atol=6e-8, rtol=2e-7)
    torch.testing.assert_close(x.grad.cpu(), (expected * (1 - expected)).float(), atol=8e-8, rtol=2e-3)
