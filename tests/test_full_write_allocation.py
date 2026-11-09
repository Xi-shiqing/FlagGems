"""Exercise output overwrite, layout, storage, stream, and training contracts."""

import pytest
import torch
import triton

import flag_gems
from flag_gems.utils import native_output_allocation as allocator
from flag_gems.utils import pointwise_dynamic

pytestmark = pytest.mark.skipif(
    not allocator.ENABLED, reason="requires FLAG_GEMS_NATIVE_FULL_WRITE=1 and its built extension"
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")])
@triton.jit
def sum_and_difference(x, y):
    return x + y, x - y


@pytest.fixture
def checked_allocator(monkeypatch):
    deterministic = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    native = allocator._empty
    calls = []

    def poisoned(shape, dtype, device):
        result = native(shape, dtype, device)
        result.fill_(float("nan") if dtype.is_floating_point else 97)
        calls.append(result)
        return result

    monkeypatch.setattr(allocator, "_empty", poisoned)
    try:
        yield calls
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.utils.deterministic.fill_uninitialized_memory
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.utils.deterministic.fill_uninitialized_memory = fill


def make_inputs(layout, dtype, device="cpu"):
    x = (torch.arange(7 * 17 * 19).reshape(7, 17, 19) % 17 - 8).to(device=device, dtype=dtype)
    if layout == "transpose":
        return x.transpose(1, 2), x.transpose(1, 2).clone()
    if layout == "slice":
        return x[:, ::2, ::2], x[:, ::2, ::2].clone()
    if layout == "broadcast":
        return x[:, :1, :], x[:1, :, :1]
    if layout == "channels_last":
        a = x.unsqueeze(0).to(memory_format=torch.channels_last)
        return a, a.clone()
    if layout == "empty":
        return x[:0], x[:0].clone()
    if layout == "scalar":
        return x[0, 0, 0], x[1, 0, 0]
    return x, x.clone()


@pytest.mark.add
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64])
@pytest.mark.parametrize(
    "layout", ["contiguous", "transpose", "slice", "broadcast", "channels_last", "empty", "scalar"]
)
def test_pointwise_overwrites_outputs(monkeypatch, checked_allocator, dtype, layout):
    a_cpu, b_cpu = make_inputs(layout, dtype)
    a, b = make_inputs(layout, dtype, flag_gems.device)
    with torch.no_grad(), flag_gems.use_gems():
        monkeypatch.setattr(allocator, "ENABLED", False)
        control = sum_and_difference(a, b)
        monkeypatch.setattr(allocator, "ENABLED", True)
        candidate = sum_and_difference(a, b)
    for old, new, expected in zip(control, candidate, (a_cpu + b_cpu, a_cpu - b_cpu)):
        assert torch.equal(new.cpu(), expected)
        assert torch.equal(old.cpu(), new.cpu())
        assert old.stride() == new.stride()
    if a.numel() and layout not in ("slice", "broadcast"):
        assert len(checked_allocator) == 2
        assert candidate[0].data_ptr() != candidate[1].data_ptr()
        assert candidate[0].data_ptr() != a.data_ptr()


@pytest.mark.add
def test_training_retains_initialized_allocation(monkeypatch, checked_allocator):
    def forbidden(*args, **kwargs):
        raise AssertionError("native output allocation reached a training operation")

    monkeypatch.setattr(allocator, "_empty", forbidden)
    x = torch.ones((17, 19), device=flag_gems.device, requires_grad=True)
    with torch.enable_grad(), flag_gems.use_gems():
        torch.add(x, x).sum().backward()
    assert torch.equal(x.grad.cpu(), torch.full((17, 19), 2.0))
    assert not checked_allocator


@pytest.mark.add
@pytest.mark.parametrize("shape", [(), (0,), (17,), (7, 19), (2, 3, 5)])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64, torch.bool, torch.complex64],
)
def test_storage_and_stream(shape, dtype):
    stream = torch.cuda.Stream()
    with torch.no_grad(), torch.cuda.stream(stream):
        a = allocator.empty(shape, dtype=dtype, device=torch.device(flag_gems.device))
        b = allocator.empty(shape, dtype=dtype, device=torch.device(flag_gems.device))
        a.fill_(1)
        b.fill_(0)
    stream.synchronize()
    assert torch.equal(a.cpu(), torch.ones(shape, dtype=dtype))
    assert torch.equal(b.cpu(), torch.zeros(shape, dtype=dtype))
    if a.numel():
        assert a.data_ptr() != b.data_ptr()


@pytest.mark.add
def test_negative_shape_rejected():
    with pytest.raises(RuntimeError):
        allocator.empty((-1,), dtype=torch.float32, device=torch.device(flag_gems.device))
