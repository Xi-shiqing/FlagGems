"""Check that clone and scan completely write their newly allocated outputs."""
import pytest
import torch

import flag_gems
from flag_gems.utils import native_output_allocation as allocator
from test_full_write_allocation import checked_allocator, make_inputs

pytestmark = pytest.mark.skipif(
    not allocator.ENABLED, reason="requires the optional native allocator"
)


@pytest.mark.clone
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64, torch.bool])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice", "channels_last", "empty", "scalar"])
@pytest.mark.parametrize("memory_format", [torch.preserve_format, torch.contiguous_format])
def test_clone_full_output(monkeypatch, checked_allocator, dtype, layout, memory_format):
    cpu, _ = make_inputs(layout, dtype)
    inp, _ = make_inputs(layout, dtype, flag_gems.device)
    with torch.no_grad():
        monkeypatch.setattr(allocator, "ENABLED", False)
        original = flag_gems.clone(inp, memory_format=memory_format)
        monkeypatch.setattr(allocator, "ENABLED", True)
        candidate = flag_gems.clone(inp, memory_format=memory_format)
    assert torch.equal(candidate.cpu(), cpu)
    assert torch.equal(candidate.cpu(), original.cpu())
    assert candidate.stride() == original.stride()
    if inp.numel():
        assert candidate.data_ptr() != inp.data_ptr()
    if layout != "slice" or memory_format == torch.contiguous_format:
        assert len(checked_allocator) == 1


@pytest.mark.cumsum
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64])
@pytest.mark.parametrize("kind", ["row", "column", "long_row", "strided", "empty"])
def test_cumsum_full_output(monkeypatch, checked_allocator, dtype, kind):
    shape = (3, 16385) if kind == "long_row" else (7, 34)
    cpu = (torch.arange(shape[0] * shape[1]).reshape(shape) % 5 - 2).to(dtype)
    inp = cpu.to(flag_gems.device)
    dim = 0 if kind == "column" else 1
    if kind == "strided":
        cpu, inp = cpu[:, ::2], inp[:, ::2]
    if kind == "empty":
        cpu, inp = cpu[:0], inp[:0]
    with torch.no_grad():
        monkeypatch.setattr(allocator, "ENABLED", False)
        original = flag_gems.cumsum(inp, dim)
        monkeypatch.setattr(allocator, "ENABLED", True)
        candidate = flag_gems.cumsum(inp, dim)
    assert torch.equal(candidate.cpu(), original.cpu())
    assert torch.equal(candidate.cpu(), torch.cumsum(cpu, dim))
    assert len(checked_allocator) >= 1


@pytest.mark.clone
def test_clone_training_keeps_original_allocation(monkeypatch, checked_allocator):
    def forbidden(*args, **kwargs):
        raise AssertionError("training clone used the inference allocator")

    monkeypatch.setattr(allocator, "_empty", forbidden)
    inp = torch.ones((17, 19), device=flag_gems.device, requires_grad=True)
    with torch.enable_grad():
        out = flag_gems.clone(inp)
        out.sum().backward()
    assert torch.equal(inp.grad.cpu(), torch.ones((17, 19)))
    assert not checked_allocator


@pytest.mark.cumsum
def test_cumsum_grad_mode_keeps_original_allocation(monkeypatch, checked_allocator):
    def forbidden(*args, **kwargs):
        raise AssertionError("grad-enabled scan used the inference allocator")

    monkeypatch.setattr(allocator, "_empty", forbidden)
    inp = torch.ones((17, 19), device=flag_gems.device)
    with torch.enable_grad():
        out = flag_gems.cumsum(inp, 1)
    assert torch.equal(out.cpu(), torch.arange(1, 20).expand(17, 19).float())
    assert not checked_allocator


@pytest.mark.cumsum
def test_cumsum_given_output_is_reused(monkeypatch, checked_allocator):
    inp = torch.ones((17, 19), device=flag_gems.device)
    out = torch.full_like(inp, float("nan"))
    with torch.no_grad():
        result = flag_gems.cumsum_out(inp, 1, out=out)
    assert result is out
    assert torch.equal(out.cpu(), torch.arange(1, 20).expand(17, 19).float())
    assert not checked_allocator
