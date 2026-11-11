"""Ordered exact coordinates, including empty/all-zero masks and fallback layouts."""
import inspect
from pathlib import Path

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="PPU nonzero")


def equal(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.nonzero
@pytest.mark.parametrize("n", [0, 1, 17, 245, 693, 2529, 8192, 8193])
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
                                  torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("pattern", ["zero", "one", "mixed"])
def test_ppu_small_nonzero_exact(n, dtype, pattern):
    cpu = ((torch.arange(n, dtype=torch.int64) % 3) - 1).to(dtype)
    if pattern == "zero":
        cpu.zero_()
    elif pattern == "one":
        cpu.fill_(1)
    ref = torch.nonzero(cpu)
    source = Path(flag_gems.__file__).resolve().parent / "runtime/backend/_thead/ops/nonzero.py"
    assert Path(inspect.getsourcefile(flag_gems.nonzero)).resolve() == source
    gpu = cpu.to(flag_gems.device)
    with torch.no_grad():
        direct = flag_gems.nonzero(gpu)
        with flag_gems.use_gems():
            actual = torch.nonzero(gpu)
    equal(direct, ref); equal(actual, ref); equal(gpu, cpu)


@pytest.mark.nonzero
@pytest.mark.parametrize("layout", ["matrix", "rank3", "rank4", "slice", "transpose", "offset", "empty", "scalar", "rank5"])
def test_ppu_small_nonzero_layout(layout):
    cpu = ((torch.arange(240, dtype=torch.int64) % 7) - 3).reshape(2, 3, 4, 10)
    gpu = cpu.to(flag_gems.device)
    if layout == "matrix":
        x = gpu.reshape(24, 10)
    elif layout == "rank3":
        x = gpu.reshape(2, 12, 10)
    elif layout == "rank4":
        x = gpu
    elif layout == "slice":
        x = gpu[..., ::2]
    elif layout == "transpose":
        x = gpu.reshape(24, 10).T
    elif layout == "offset":
        x = gpu.reshape(-1)[5:206]
    elif layout == "empty":
        x = gpu[:0]
    elif layout == "scalar":
        x = gpu.reshape(-1)[0].reshape(())
    else:
        x = gpu.reshape(1, 2, 3, 4, 10)
    ref = torch.nonzero(x.cpu())
    with torch.no_grad(), flag_gems.use_gems():
        actual = torch.nonzero(x)
    equal(actual, ref)


@pytest.mark.nonzero
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_ppu_small_nonzero_floating_special_values(dtype):
    cpu = torch.tensor([0.0, -0.0, float('nan'), float('inf'), -float('inf'),
                        torch.finfo(dtype).tiny / 2], dtype=dtype)
    ref = torch.nonzero(cpu)
    with torch.no_grad(), flag_gems.use_gems():
        actual = torch.nonzero(cpu.to(flag_gems.device))
    equal(actual, ref)


@pytest.mark.nonzero
def test_ppu_small_nonzero_tuple_columns():
    cpu = torch.tensor([[0, 3, 0], [2, 0, -1]], dtype=torch.int64)
    ref = torch.nonzero(cpu, as_tuple=True)
    gpu = cpu.to(flag_gems.device)
    with torch.no_grad():
        actual = flag_gems.nonzero(gpu, as_tuple=True)
    assert len(actual) == len(ref)
    for a, b in zip(actual, ref):
        equal(a, b)
