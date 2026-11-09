"""Full overwrite and layout checks for PPU copy output allocations."""
import importlib
import types

import pytest
import torch

import flag_gems
from flag_gems.utils import native_output_allocation as allocator

pytestmark = pytest.mark.skipif(
    not allocator.ENABLED, reason="Requires the optional full-write allocator"
)


def bits(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def operand(layout):
    base = torch.arange(217, device=flag_gems.device, dtype=torch.float32).reshape(7, 31) / 8
    return {
        "contiguous": lambda: base,
        "transpose": lambda: base.T,
        "sliced": lambda: base[:, ::2],
        "expanded": lambda: base[:1].expand(9, 31),
        "empty": lambda: base[:0],
        "scalar": lambda: base[0, 0],
        "channels_last": lambda: base.reshape(1, 7, 1, 31).contiguous(
            memory_format=torch.channels_last
        ),
    }[layout]()


@pytest.fixture
def poison(monkeypatch):
    original = allocator._empty
    count = []
    previous = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    def filled(shape, dtype, device):
        value = original(shape, dtype, device)
        value.fill_(float("nan") if dtype.is_floating_point else 97)
        count.append((tuple(shape), dtype))
        return value

    monkeypatch.setattr(allocator, "_empty", filled)
    try:
        yield count
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.utils.deterministic.fill_uninitialized_memory = fill


@pytest.mark.to_copy
@pytest.mark.parametrize(
    "layout",
    ["contiguous", "transpose", "sliced", "expanded", "empty", "scalar", "channels_last"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64, torch.bool])
@pytest.mark.parametrize("memory_format", [torch.preserve_format, torch.contiguous_format])
def test_to_copy_overwrite(poison, monkeypatch, layout, dtype, memory_format):
    module = importlib.import_module("flag_gems.ops.to")
    x = operand(layout)
    values = []
    with torch.no_grad():
        for enabled in [False, True]:
            monkeypatch.setattr(
                module,
                "_native_output",
                allocator if enabled else types.SimpleNamespace(ENABLED=False),
            )
            values.append(flag_gems.to_copy(x, dtype=dtype, memory_format=memory_format))
    assert torch.equal(bits(values[0]), bits(values[1]))
    assert values[0].stride() == values[1].stride()
    assert values[1].shape == x.shape and values[1].dtype == dtype
    if x.numel():
        assert values[1].data_ptr() != x.data_ptr()
    expected = x.cpu().to(dtype=dtype, copy=True)
    assert torch.equal(bits(values[1]), bits(expected))


@pytest.mark.to_copy
@pytest.mark.lift_fresh
def test_gradient_mode_and_special_values(poison, monkeypatch):
    module = importlib.import_module("flag_gems.ops.to")
    monkeypatch.setattr(module, "_native_output", allocator)
    x = torch.tensor([float("nan"), float("inf"), -0.0, 1.0], device=flag_gems.device)
    with torch.enable_grad():
        a, b = flag_gems.to_copy(x), flag_gems.lift_fresh(x)
    assert not poison
    assert torch.equal(bits(a), bits(x)) and torch.equal(bits(b), bits(x))
    with torch.no_grad():
        a, b = flag_gems.to_copy(x), flag_gems.lift_fresh(x)
    assert len(poison) == 1
    assert torch.equal(bits(a), bits(x)) and torch.equal(bits(b), bits(x))


@pytest.mark.to_copy
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64],
)
def test_to_copy_boolean_truth_values(dtype):
    values = [0, 1, -1, 2, 256, -256]
    if dtype.is_floating_point:
        values += [0.125, -0.125, -0.0, float("inf"), -float("inf"), float("nan")]
    x = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    expected = x.cpu().to(torch.bool)
    with torch.no_grad(), flag_gems.use_gems():
        actual = x.to(torch.bool)
    assert torch.equal(bits(actual), bits(expected))
