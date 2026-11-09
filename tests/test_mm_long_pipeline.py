"""Registered long-K MM, output layouts, autograd and CPU FP64 samples."""
import importlib

import pytest
import torch

import flag_gems
from flag_gems.runtime import device as runtime_device


def _operands(m, n, k, transpose=True):
    g = torch.Generator(device="cpu").manual_seed(20261107)
    a = torch.randn((k, m) if transpose else (m, k), generator=g)
    return a.t() if transpose else a, torch.randn((k, n), generator=g)


def _observe(monkeypatch):
    project = flag_gems.mm.__globals__["_project_mm"]
    helper = importlib.import_module(project.__module__.rsplit(".", 1)[0]+".mm_long_pipeline")
    original = helper.try_mm
    hits = []

    def checked(*args):
        value = original(*args)
        if value is not None:
            hits.append((tuple(args[0].shape), tuple(args[1].shape)))
        return value

    monkeypatch.setattr(helper, "try_mm", checked)
    return hits


@pytest.mark.mm
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU long-K pipeline")
@pytest.mark.parametrize("tf32", [False, True])
@pytest.mark.parametrize("shape", [(65, 65, 8209), (257, 129, 16387)])
@pytest.mark.parametrize("transpose_out", [False, True])
def test_registered_long_pipeline_out(tf32, shape, transpose_out, monkeypatch):
    m, n, k = shape
    ah, bh = _operands(m, n, k)
    a, b = ah.to(flag_gems.device), bh.to(flag_gems.device)
    out = torch.empty((n, m) if transpose_out else (m, n), device=a.device)
    if transpose_out:
        out = out.t()
    hits = _observe(monkeypatch)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        with flag_gems.use_gems():
            monkeypatch.setenv("FLAG_GEMS_PPU_LONG_MM_PIPELINE", "0")
            expected = torch.mm(a, b)
            monkeypatch.setenv("FLAG_GEMS_PPU_LONG_MM_PIPELINE", "1")
            actual = torch.ops.aten.mm.out(a, b, out=out)
        assert hits and actual.data_ptr() == out.data_ptr()
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        reference = ah[:16].double() @ bh[:, :16].double()
        error = (actual[:16, :16].cpu().double()-reference).norm()/reference.norm()
        assert float(error) < (0.002 if tf32 else 2e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.mm
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU long-K boundary")
@pytest.mark.parametrize("tf32", [False, True])
def test_registered_long_pipeline_8192_fallback(tf32, monkeypatch):
    ah, bh = _operands(65, 65, 8192)
    a, b = ah.to(flag_gems.device), bh.to(flag_gems.device)
    hits = _observe(monkeypatch)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        with flag_gems.use_gems():
            monkeypatch.setenv("FLAG_GEMS_PPU_LONG_MM_PIPELINE", "0")
            expected = torch.mm(a, b)
            monkeypatch.setenv("FLAG_GEMS_PPU_LONG_MM_PIPELINE", "1")
            actual = torch.mm(a, b)
        assert not hits
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.mm
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU long-K backward")
@pytest.mark.parametrize("tf32", [False, True])
def test_registered_backward_weight_long_pipeline(tf32, monkeypatch):
    generator = torch.Generator().manual_seed(20261109)
    ah = torch.randn((65539, 65), generator=generator)
    bh = torch.randn((65, 65), generator=generator)
    dy = torch.randn((65539, 65), generator=generator)
    hits = _observe(monkeypatch)
    previous = torch.backends.cuda.matmul.allow_tf32
    values = []
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        for switch in ["0", "1"]:
            a = ah.to(flag_gems.device).requires_grad_()
            b = bh.to(flag_gems.device).requires_grad_()
            monkeypatch.setenv("FLAG_GEMS_PPU_LONG_MM_PIPELINE", switch)
            with flag_gems.use_gems():
                out = torch.mm(a, b)
                out.backward(dy.to(a.device))
            values.append(tuple(v.detach().cpu() for v in (out, a.grad, b.grad)))
        assert ((65, 65539), (65539, 65)) in hits
        for actual, expected in zip(values[1], values[0]):
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        reference = ah[:, :16].double().t() @ dy[:, :16].double()
        error = (values[1][2][:16, :16].double()-reference).norm()/reference.norm()
        assert float(error) < (0.002 if tf32 else 2e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.mm
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU long-K pipeline")
@pytest.mark.parametrize("tf32", [False, True])
@pytest.mark.parametrize("transpose_a", [False, True])
def test_registered_long_pipeline_grad(tf32, transpose_a, monkeypatch):
    ah, bh = _operands(65, 65, 8209, transpose_a)
    dy = torch.randn((65, 65), generator=torch.Generator().manual_seed(20261108))
    previous = torch.backends.cuda.matmul.allow_tf32
    hits = _observe(monkeypatch)
    values = []
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        for switch in ["0", "1"]:
            a = ah.to(flag_gems.device).requires_grad_()
            b = bh.to(flag_gems.device).requires_grad_()
            monkeypatch.setenv("FLAG_GEMS_PPU_LONG_MM_PIPELINE", switch)
            with flag_gems.use_gems():
                value = torch.mm(a, b)
                value.backward(dy.to(a.device))
            values.append(tuple(v.detach().cpu() for v in (value, a.grad, b.grad)))
        assert bool(hits) == transpose_a
        for actual, expected in zip(values[1], values[0]):
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        references = [dy[:16].double() @ bh.double().t(), ah[:, :16].double().t() @ dy[:, :16].double()]
        for actual, reference in zip([values[1][1][:16], values[1][2][:16, :16]], references):
            error = (actual.double()-reference).norm()/reference.norm()
            assert float(error) < (0.002 if tf32 else 2e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
