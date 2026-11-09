"""Check complete writes and unchanged results at registered reduction APIs."""
import importlib
import types

import pytest
import torch

import flag_gems
from flag_gems.utils import native_output_allocation as allocator

pytestmark = [
    pytest.mark.sum,
    pytest.mark.mean,
    pytest.mark.any,
    pytest.mark.new_ones,
    pytest.mark.skipif(not allocator.ENABLED, reason="Requires full-write allocator"),
]
MODULES = [
    importlib.import_module("flag_gems.ops." + n)
    for n in ["sum", "mean", "any", "new_ones"]
]


def bits(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def select(monkeypatch, enabled):
    for module in MODULES:
        monkeypatch.setattr(
            module,
            "_native_output",
            allocator if enabled else types.SimpleNamespace(empty_for_overwrite=torch.empty),
        )


@pytest.fixture
def poisoned(monkeypatch):
    original = allocator._empty
    seen = []
    deterministic = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    def allocate(shape, dtype, device):
        result = original(shape, dtype, device)
        result.fill_(float("nan") if dtype.is_floating_point else 97)
        seen.append((tuple(shape), dtype))
        return result

    monkeypatch.setattr(allocator, "_empty", allocate)
    try:
        yield seen
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.utils.deterministic.fill_uninitialized_memory = fill


@pytest.mark.parametrize("operator", ["sum", "mean", "any"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "layout", ["contiguous", "transpose", "sliced", "expanded", "empty"]
)
def test_reduction_complete_writes(monkeypatch, poisoned, operator, dtype, layout):
    # Exactly represented fractions also check half-precision intermediate stores.
    base = ((torch.arange(5 * 7 * 13).reshape(5, 7, 13) % 31) - 15).to(dtype) / 32
    transform = {
        "contiguous": lambda t: t,
        "transpose": lambda t: t.transpose(0, 2),
        "sliced": lambda t: t[:, ::2, ::2],
        "expanded": lambda t: t[:1].expand(9, 7, 13),
        "empty": lambda t: t[:0],
    }[layout]
    cpu = transform(base)
    x = transform(base.to(flag_gems.device))
    function = getattr(torch, operator)
    for dims in [None, (-1,), (0, 2)]:
        for keepdim in [False, True]:
            kwargs = {"dim": dims, "keepdim": keepdim}
            expected = function(cpu, **kwargs)
            values = []
            with torch.no_grad():
                for enabled in [False, True]:
                    select(monkeypatch, enabled)
                    with flag_gems.use_gems():
                        values.append(function(x, **kwargs))
            assert values[0].shape == values[1].shape == expected.shape
            assert values[0].dtype == values[1].dtype == expected.dtype
            assert torch.equal(bits(values[0]), bits(values[1]))
            if expected.is_floating_point():
                torch.testing.assert_close(values[1].cpu(), expected, equal_nan=True)
            else:
                assert torch.equal(values[1].cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64])
def test_integer_and_boolean_reductions(monkeypatch, poisoned, dtype):
    cpu = (torch.arange(7 * 13).reshape(7, 13) % 5).to(dtype)
    x = cpu.to(flag_gems.device)
    for operator in ["sum", "any"]:
        function = getattr(torch, operator)
        for dim in [None, 0, -1]:
            for enabled in [False, True]:
                select(monkeypatch, enabled)
                with torch.no_grad(), flag_gems.use_gems():
                    value = function(x, dim=dim)
                expected = function(cpu, dim=dim)
                assert value.dtype == expected.dtype
                assert torch.equal(value.cpu(), expected)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool]
)
@pytest.mark.parametrize("shape", [(), (0, 7), (7, 31)])
def test_new_ones_complete_writes(monkeypatch, poisoned, dtype, shape):
    x = torch.empty(3, device=flag_gems.device)
    for enabled in [False, True]:
        select(monkeypatch, enabled)
        with torch.no_grad(), flag_gems.use_gems():
            value = x.new_ones(shape, dtype=dtype)
        assert value.shape == shape and value.dtype == dtype and not value.requires_grad
        assert torch.equal(value.cpu(), torch.ones(shape, dtype=dtype))


def test_out_parameter_and_backward(monkeypatch, poisoned):
    gradients = {False: [], True: []}
    for enabled in [False, True]:
        select(monkeypatch, enabled)
        for name in ["sum", "mean"]:
            x = (
                torch.arange(91, dtype=torch.float32, device=flag_gems.device)
                .reshape(7, 13)
                .requires_grad_()
            )
            with flag_gems.use_gems():
                y = getattr(torch, name)(x, dim=(0, 1))
            (gradient,) = torch.autograd.grad(y, x)
            expected = torch.full((7, 13), 1.0 if name == "sum" else 1.0 / 91)
            assert torch.equal(gradient.cpu(), expected)
            gradients[enabled].append(gradient.cpu())
        x = torch.arange(91, device=flag_gems.device).reshape(7, 13)
        out = torch.empty(7, device=flag_gems.device, dtype=x.dtype)
        pointer = out.data_ptr()
        with torch.no_grad(), flag_gems.use_gems():
            actual = torch.sum(x, dim=1, out=out)
        assert actual is out and out.data_ptr() == pointer
        assert torch.equal(actual.cpu(), torch.arange(91).reshape(7, 13).sum(1))
    assert all(torch.equal(a, b) for a, b in zip(gradients[False], gradients[True]))


def test_special_values_and_random_fp32(monkeypatch, poisoned):
    generator = torch.Generator().manual_seed(20261109)
    for cpu in [
        torch.randn((13, 77), generator=generator),
        torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0]),
    ]:
        x = cpu.to(flag_gems.device)
        for operator in ["sum", "mean", "any"]:
            values = []
            for enabled in [False, True]:
                select(monkeypatch, enabled)
                with torch.no_grad(), flag_gems.use_gems():
                    values.append(getattr(torch, operator)(x))
            assert torch.equal(bits(values[0]), bits(values[1]))
            torch.testing.assert_close(values[1].cpu(), getattr(torch, operator)(cpu), equal_nan=True)


@pytest.mark.parametrize("layout", ["transpose", "sliced", "expanded"])
def test_any_strided_all_false(monkeypatch, poisoned, layout):
    base = torch.zeros((7, 13), device=flag_gems.device)
    x = {"transpose": lambda: base.T, "sliced": lambda: base[:, ::2],
         "expanded": lambda: base[:1].expand(27, 13)}[layout]()
    assert not x.is_contiguous()
    for enabled in [False, True]:
        select(monkeypatch, enabled)
        with torch.no_grad(), flag_gems.use_gems():
            result = torch.any(x)
        assert not result.item()


def test_scalar_sum_out_strided(monkeypatch, poisoned):
    base = torch.arange(64, device=flag_gems.device, dtype=torch.int64) % 2
    x = base[::2]
    assert not x.is_contiguous()
    for enabled in [False, True]:
        select(monkeypatch, enabled)
        out = torch.empty((), device=x.device, dtype=x.dtype)
        pointer = out.data_ptr()
        with torch.no_grad(), flag_gems.use_gems():
            result = torch.ops.aten.sum.out(x, out=out)
        assert result is out and out.data_ptr() == pointer
        assert result.item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_any_dim_truth_values(monkeypatch, poisoned, dtype):
    tiny = torch.finfo(dtype).tiny / 2
    cpu = torch.tensor(
        [[0.0, -0.0], [tiny, 0], [-tiny, 0], [float("nan"), 0],
         [float("inf"), 0], [-float("inf"), 0], [0.125, 0]], dtype=dtype,
    )
    x = cpu.to(flag_gems.device)
    expected = cpu.any(dim=1)
    for enabled in [False, True]:
        select(monkeypatch, enabled)
        with torch.no_grad(), flag_gems.use_gems():
            value = x.any(dim=1)
        assert torch.equal(value.cpu(), expected)
        for row, reference in zip(x.unbind(), expected):
            with torch.no_grad(), flag_gems.use_gems():
                scalar = row.any()
            assert scalar.item() == reference.item()


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
@pytest.mark.parametrize("length", [1, 511, 512, 513, 1023, 1024, 1025])
def test_any_single_block_boundaries(monkeypatch, poisoned, dtype, length):
    x = torch.zeros(length, dtype=dtype, device=flag_gems.device)
    for nonzero in [False, True]:
        x[-1] = nonzero
        expected = x.cpu().any()
        for enabled in [False, True]:
            select(monkeypatch, enabled)
            with torch.no_grad(), flag_gems.use_gems():
                result = x.any()
            assert result.dtype == torch.bool and result.shape == ()
            assert result.item() == expected.item()
