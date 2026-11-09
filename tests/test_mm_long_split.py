"""Public MM/out long-K partitioning and fixed reduction order."""

import importlib

import pytest
import torch

import flag_gems
from flag_gems.runtime import device as runtime_device

pytestmark = pytest.mark.skipif(
    runtime_device.vendor_name != "thead", reason="PPU long-K partitioning"
)


def helper():
    project = flag_gems.mm.__globals__["_project_mm"]
    return importlib.import_module(project.__module__.rsplit(".", 1)[0] + ".mm_long_split")


def operands(m, n, k):
    generator = torch.Generator().manual_seed(20261109)
    return torch.randn((k, m), generator=generator).t(), torch.randn(
        (k, n), generator=generator
    )


@pytest.fixture(params=[False, True], ids=["fp32", "tf32"])
def precision(request, monkeypatch):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = request.param
    monkeypatch.setenv("FLAG_GEMS_PPU_LONG_MM_PIPELINE", "1")
    monkeypatch.setenv("FLAG_GEMS_PPU_LONG_MM_SPLIT", "1")
    try:
        yield request.param
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.mm
@pytest.mark.parametrize("shape", [(33, 65, 8209), (65, 129, 65539), (129, 257, 131071)])
@pytest.mark.parametrize("transpose_out", [False, True])
def test_long_split_registered_out(precision, shape, transpose_out, monkeypatch):
    ah, bh = operands(*shape)
    a, b = ah.to(flag_gems.device), bh.to(flag_gems.device)
    m, n, _ = shape
    out = torch.empty((n, m) if transpose_out else (m, n), device=a.device)
    if transpose_out:
        out = out.t()
    module = helper()
    original = module.try_mm
    hits = []

    def observed(*args):
        value = original(*args)
        if value is not None:
            hits.append(1)
        return value

    monkeypatch.setattr(module, "try_mm", observed)
    result = flag_gems.mm_out(a, b, out=out)
    assert hits and result is out
    reference = ah.double() @ bh.double()
    relative = (result.cpu().double() - reference).norm() / reference.norm()
    assert float(relative) < (0.002 if precision else 2e-5)


@pytest.mark.mm
@pytest.mark.parametrize("groups", [1, 4, 16, 64])
def test_partition_order_and_precision_policy(precision, groups):
    ah, bh = operands(33, 65, 10003)
    a, b = ah.to(flag_gems.device), bh.to(flag_gems.device)
    module = helper()
    reference = module._execute(a, b, None, precision, 64, 64, 1)
    torch.backends.cuda.matmul.allow_tf32 = not precision
    actual = module._execute(a, b, None, precision, 64, groups, 3)
    assert torch.equal(actual.view(torch.int32), reference.view(torch.int32))


@pytest.mark.mm
@pytest.mark.parametrize("k", [0, 1, 8192])
def test_existing_boundary_unchanged(precision, k, monkeypatch):
    ah, bh = operands(33, 65, k)
    a, b = ah.to(flag_gems.device), bh.to(flag_gems.device)
    values = []
    for enabled in ["0", "1"]:
        monkeypatch.setenv("FLAG_GEMS_PPU_LONG_MM_SPLIT", enabled)
        values.append(flag_gems.mm(a, b))
    assert torch.equal(values[0].view(torch.int32), values[1].view(torch.int32))


@pytest.mark.mm
def test_plan_work_and_workspace_bounds():
    module = helper()
    for m, n, k in [(32, 32, 8193), (65, 257, 100003), (512, 512, 1228800)]:
        splits, groups, stages, tile = module._plan(m, n, k, 64)
        assert 2 <= groups <= splits <= 64
        assert splits * m * n * 4 <= 64 * 1024**2 and k // splits >= 128
        assert stages == 3 and tile == 64
    assert module._plan(64, 64, 8192, 64) is None
    assert module._plan(1024, 1024, 100003, 64) is None
