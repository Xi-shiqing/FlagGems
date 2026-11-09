"""Check the registered ordered MM against rollback and CPU FP64 arithmetic."""
import pytest
import torch

import flag_gems
from flag_gems.runtime import device as runtime_device


def _inputs(transposed):
    generator = torch.Generator(device="cpu").manual_seed(20261107)
    a = torch.randn((17001, 255), generator=generator)
    b = torch.randn((383, 255) if transposed else (255, 383), generator=generator)
    return a, b.t() if transposed else b


@pytest.mark.mm
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU ordered MM")
@pytest.mark.parametrize("tf32", [False, True])
@pytest.mark.parametrize("rhs_transposed", [False, True])
@pytest.mark.parametrize("output_transposed", [False, True])
def test_registered_ordered_mm_out(tf32, rhs_transposed, output_transposed, monkeypatch):
    cpu_a, cpu_b = _inputs(rhs_transposed)
    a, b = cpu_a.to(flag_gems.device), cpu_b.to(flag_gems.device)
    shape = (383, 17001) if output_transposed else (17001, 383)
    out = torch.empty(shape, dtype=a.dtype, device=a.device)
    if output_transposed:
        out = out.t()
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        with flag_gems.use_gems():
            monkeypatch.setenv("FLAG_GEMS_PPU_ORDERED_DIRECT_MM", "0")
            expected = torch.mm(a, b)
            monkeypatch.setenv("FLAG_GEMS_PPU_ORDERED_DIRECT_MM", "1")
            actual = torch.ops.aten.mm.out(a, b, out=out)
        assert actual.data_ptr() == out.data_ptr()
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        reference = cpu_a[:16].double() @ cpu_b[:, :16].double()
        error = (actual[:16, :16].cpu().double() - reference).norm() / reference.norm()
        assert float(error) < (0.002 if tf32 else 2e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.mm
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU ordered MM")
@pytest.mark.parametrize("tf32", [False, True])
@pytest.mark.parametrize("rhs_transposed", [False, True])
def test_registered_ordered_mm_autograd(tf32, rhs_transposed, monkeypatch):
    cpu_a, cpu_b = _inputs(rhs_transposed)
    dy = torch.randn((17001, 383), generator=torch.Generator().manual_seed(20261108))
    previous = torch.backends.cuda.matmul.allow_tf32
    results = []
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        for enabled in ["0", "1"]:
            a = cpu_a.to(flag_gems.device).requires_grad_()
            b = cpu_b.to(flag_gems.device).requires_grad_()
            monkeypatch.setenv("FLAG_GEMS_PPU_ORDERED_DIRECT_MM", enabled)
            with flag_gems.use_gems():
                output = torch.mm(a, b)
                output.backward(dy.to(a.device))
            results.append(tuple(t.detach().cpu() for t in (output, a.grad, b.grad)))
        for actual, expected in zip(results[1], results[0]):
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        ref_a = dy[:16].double() @ cpu_b.double().t()
        ref_b = cpu_a[:, :16].double().t() @ dy[:, :16].double()
        for actual, reference in [(results[1][1][:16], ref_a), (results[1][2][:16, :16], ref_b)]:
            error = (actual.double() - reference).norm() / reference.norm()
            assert float(error) < (0.002 if tf32 else 2e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
