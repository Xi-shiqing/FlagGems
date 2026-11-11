"""Exact stable values/indices and unique outputs through the real PPU API."""
import inspect
from pathlib import Path

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="PPU integer sort")
DTYPES = [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
LENGTHS = [0, 1, 2, 7, 31, 129, 245, 693, 1023, 2049, 4096, 4097]


def values(n, dtype):
    if dtype == torch.bool:
        pattern = [False, True, True, False, False, True, False]
    else:
        info = torch.iinfo(dtype)
        pattern = [0, info.min, info.max, 0, 1, info.max, info.min]
        if info.min < 0:
            pattern += [-1, -1, -7]
        if dtype == torch.int64:
            pattern += [2**60 + 3, 2**60 + 1, -(2**60 + 1)]
    return torch.tensor((pattern * ((n + len(pattern) - 1) // len(pattern)))[:n], dtype=dtype)


def assert_equal(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.sort
@pytest.mark.sort_stable
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("descending", [False, True])
def test_ppu_short_integer_sort_exact(dtype, n, descending):
    source = Path(flag_gems.__file__).resolve().parent / "runtime/backend/_thead/ops/sort.py"
    assert Path(inspect.getsourcefile(flag_gems.sort)).resolve() == source
    cpu = values(n, dtype)
    ref = torch.sort(cpu, stable=True, descending=descending)
    gpu = cpu.to(flag_gems.device)
    with torch.no_grad():
        direct = flag_gems.sort(gpu, descending=descending)
        with flag_gems.use_gems():
            stable = torch.sort(gpu, stable=True, descending=descending)
            unstable = torch.sort(gpu, stable=False, descending=descending)
    for result in (direct, stable, unstable):
        assert_equal(result[0], ref[0])
        assert_equal(result[1], ref[1])
    assert_equal(gpu, cpu)


@pytest.mark.sort
@pytest.mark.parametrize("layout", ["matrix", "transpose", "slice", "expand", "offset", "scalar", "empty_axis", "rank3"])
@pytest.mark.parametrize("descending", [False, True])
def test_ppu_short_integer_sort_layout(layout, descending):
    base = values(3 * 34 + 5, torch.int64).to(flag_gems.device)
    if layout == "matrix":
        x, dim = base[:102].reshape(3, 34), 0
    elif layout == "transpose":
        x, dim = base[:102].reshape(3, 34).T, 0
    elif layout == "slice":
        x, dim = base[:102].reshape(3, 34)[:, ::2], -1
    elif layout == "expand":
        x, dim = base[:17].reshape(1, 17).expand(3, 17), 0
    elif layout == "offset":
        x, dim = base[5:38], -1
    elif layout == "scalar":
        x, dim = base[0].reshape(()), -1
    elif layout == "empty_axis":
        x, dim = base[:0].reshape(3, 0), -1
    else:
        x, dim = base[:102].reshape(2, 3, 17), 1
    before = x.cpu().clone()
    ref = torch.sort(before, dim=dim, stable=True, descending=descending)
    with torch.no_grad(), flag_gems.use_gems():
        actual = torch.sort(x, dim=dim, stable=True, descending=descending)
    assert_equal(actual[0], ref[0]); assert_equal(actual[1], ref[1]); assert_equal(x, before)


@pytest.mark.unique2
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("n", [0, 1, 17, 245, 693, 2529, 4097])
@pytest.mark.parametrize("inverse,counts", [(False, False), (True, False), (False, True), (True, True)])
def test_ppu_short_sort_unique(dtype, n, inverse, counts):
    cpu = values(n, dtype)
    ref = torch.ops.aten._unique2(cpu, True, inverse, counts)
    with torch.no_grad(), flag_gems.use_gems():
        actual = torch.ops.aten._unique2(cpu.to(flag_gems.device), True, inverse, counts)
    for x, y in zip(actual, ref):
        assert_equal(x, y)


@pytest.mark.sort
def test_ppu_sort_float_fallback():
    cpu = torch.tensor([[2.0, -1.0, 2.0, 0.0, -4.0]], dtype=torch.float32)
    for descending in (False, True):
        ref = torch.sort(cpu, stable=True, descending=descending)
        with torch.no_grad(), flag_gems.use_gems():
            actual = torch.sort(cpu.to(flag_gems.device), stable=True, descending=descending)
        assert_equal(actual[0], ref[0]); assert_equal(actual[1], ref[1])
