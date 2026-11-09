import pytest
import torch
import torch.nn.functional as F

import flag_gems
from flag_gems.runtime import device as runtime_device


@pytest.mark.silu
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU SiLU rounding")
@pytest.mark.parametrize("shape", [(), (0,), (1031,), (17, 31), (4097, 127)])
@pytest.mark.parametrize("tf32", [False, True])
def test_ppu_silu_against_double(shape, tf32):
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        generator = torch.Generator().manual_seed(8026)
        cpu = torch.randn(shape, generator=generator) * 12
        x = cpu.to(flag_gems.device)
        with torch.no_grad(), flag_gems.use_gems():
            actual = F.silu(x)
        torch.testing.assert_close(actual.cpu(), F.silu(cpu.double()).float(), atol=1e-7, rtol=3e-7)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.silu
@pytest.mark.silu_backward
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU SiLU rounding")
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("layout", ["transpose", "slice", "contiguous"])
def test_ppu_silu_layout_and_backward(layout, inplace):
    cpu = torch.linspace(-12, 12, 17 * 31).reshape(17, 31)
    x = cpu.to(flag_gems.device)
    if layout == "transpose":
        cpu, x = cpu.t(), x.t()
    elif layout == "slice":
        cpu, x = cpu[:, ::2], x[:, ::2]
    x = x.detach().requires_grad_(True)
    with flag_gems.use_gems():
        y = F.silu(x.clone(), inplace=inplace)
        y.sum().backward()
    reference = cpu.double().requires_grad_(True)
    expected = F.silu(reference)
    expected.sum().backward()
    torch.testing.assert_close(y.cpu(), expected.float(), atol=1e-7, rtol=3e-7)
    torch.testing.assert_close(x.grad.cpu(), reference.grad.float(), atol=8e-7, rtol=1e-5)


@pytest.mark.silu
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU SiLU rounding")
@pytest.mark.parametrize("inplace", [False, True])
def test_ppu_silu_flat_rounding(monkeypatch, inplace):
    cpu = torch.linspace(-40, 40, 262147)
    x = cpu.to(flag_gems.device)
    with torch.no_grad(), flag_gems.use_gems():
        monkeypatch.setenv("FLAG_GEMS_PPU_SILU_HOT", "0")
        reference = F.silu(x)
        monkeypatch.setenv("FLAG_GEMS_PPU_SILU_HOT", "1")
        actual = F.silu(x, inplace=inplace)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
