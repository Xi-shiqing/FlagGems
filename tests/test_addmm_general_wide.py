"""Check registered addmm, compiled tiles, gradients, output layouts and streams."""
import importlib
from contextlib import contextmanager

import pytest
import torch

import flag_gems


def backend():
    return importlib.import_module(flag_gems.addmm.__globals__["_project_addmm"].__module__)


def operands(m, n, k, transpose_rhs=True):
    generator = torch.Generator().manual_seed(20261110)
    ah = torch.randn((m, k), generator=generator)
    bh = torch.randn((n, k), generator=generator).t() if transpose_rhs else torch.randn((k, n), generator=generator)
    bias_h = torch.randn((n,), generator=generator)
    a = ah.to(flag_gems.device)
    b = torch.empty_strided(bh.shape, bh.stride(), device=flag_gems.device)
    b.copy_(bh)
    return a, b, bias_h.to(flag_gems.device), ah, bh, bias_h


def same_bits(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@contextmanager
def precision(tf32):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.addmm
@pytest.mark.parametrize("shape", [(1225, 768, 384), (2725, 768, 768), (430, 768, 384), (127, 139, 259)])
@pytest.mark.parametrize("transpose_rhs", [False, True])
@pytest.mark.parametrize("tf32", [False, True])
def test_registered_wide_addmm(monkeypatch, shape, transpose_rhs, tf32):
    module = backend()
    m, n, k = shape
    a, b, bias, ah, bh, bias_h = operands(m, n, k, transpose_rhs)
    values = []
    with precision(tf32), torch.no_grad(), flag_gems.use_gems():
        for enabled in [False, True, False, True]:
            monkeypatch.setattr(module, "_WIDE_ENABLED", enabled)
            values.append(torch.addmm(bias, a, b))
        storage = torch.full((m, 2 * n), 123.5, device=a.device)
        output = storage[:, ::2]
        returned = torch.addmm(bias, a, b, out=output)
    assert returned is output
    assert (storage[:, 1::2] == 123.5).all()
    for value in values[1:] + [output]:
        same_bits(values[0], value)
    reference = ah[:16].double() @ bh[:, :16].double() + bias_h[:16].double()
    torch.testing.assert_close(
        values[0][:16, :16].cpu().double(), reference,
        rtol=2e-3 if tf32 else 2e-5, atol=0.04 if tf32 else 6e-5,
    )


@pytest.mark.addmm
@pytest.mark.parametrize("alpha,beta,bias_shape", [(0.25, -0.5, "column"), (0.0, 1.0, "scalar"), (1.0, 0.0, "vector")])
@pytest.mark.parametrize("tf32", [False, True])
def test_wide_addmm_coefficients_and_bias(monkeypatch, alpha, beta, bias_shape, tf32):
    module = backend()
    a, b, bias, ah, bh, bias_h = operands(1225, 768, 384)
    if bias_shape == "column":
        bias_h = torch.linspace(-1, 1, 1225).unsqueeze(1)
    elif bias_shape == "scalar":
        bias_h = torch.tensor(0.25)
    bias = bias_h.to(a.device)
    values = []
    with precision(tf32), torch.no_grad(), flag_gems.use_gems():
        for enabled in [False, True, False, True]:
            monkeypatch.setattr(module, "_WIDE_ENABLED", enabled)
            values.append(torch.addmm(bias, a, b, alpha=alpha, beta=beta))
            output = torch.empty_like(values[-1])
            torch.addmm(bias, a, b, out=output, alpha=alpha, beta=beta)
            same_bits(values[-1], output)
    for value in values[1:]:
        same_bits(values[0], value)
    reference = (ah[:16].double() @ bh[:, :16].double()) * alpha
    reference += bias_h.broadcast_to((1225, 768))[:16, :16].double() * beta
    torch.testing.assert_close(
        values[0][:16, :16].cpu().double(), reference,
        rtol=2e-3 if tf32 else 2e-5, atol=0.04 if tf32 else 6e-5,
    )


@pytest.mark.addmm
@pytest.mark.parametrize("tf32", [False, True])
def test_wide_addmm_actual_tiles_and_resource_tail(monkeypatch, tf32):
    module = backend()
    sms = module._general_device_sms(torch.device(flag_gems.device))
    original = module.addmm_kernel
    observed = []

    class Kernel:
        def __getitem__(self, grid):
            launch = original[grid]

            def call(*args, **kwargs):
                returned = launch(*args, **kwargs)
                observed.append(tuple(int(returned[1][k]) for k in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"]))
                return returned

            return call

    monkeypatch.setattr(module, "addmm_kernel", Kernel())
    partial = next(x for x in range(1, 2 * sms) if 6 * x >= sms and 4 * 6 * x < 3 * ((6 * x + sms - 1) // sms) * sms)
    with precision(tf32), torch.no_grad(), flag_gems.use_gems():
        for m, resource_ok in [(sms * 64, True), (partial * 64, False)]:
            a, b, bias, *_ = operands(m, 768, 128)
            values = []
            for enabled in [False, True, False, True]:
                monkeypatch.setattr(module, "_WIDE_ENABLED", enabled)
                values.append(torch.addmm(bias, a, b))
                expected = (64, 128, 32) if enabled and not tf32 and resource_ok else (32, 32, 32)
                assert observed[-1] == expected
            for value in values[1:]:
                same_bits(values[0], value)


@pytest.mark.addmm
@pytest.mark.parametrize("tf32", [False, True])
def test_wide_addmm_backward(monkeypatch, tf32):
    module = backend()
    _, _, _, ah, bh, bias_h = operands(1225, 768, 384)
    values = []
    with precision(tf32), flag_gems.use_gems():
        for enabled in [False, True]:
            monkeypatch.setattr(module, "_WIDE_ENABLED", enabled)
            a = ah.to(flag_gems.device).requires_grad_()
            b = torch.empty_strided(bh.shape, bh.stride(), device=flag_gems.device)
            b.copy_(bh)
            b.requires_grad_()
            bias = bias_h.to(flag_gems.device).requires_grad_()
            result = torch.addmm(bias, a, b)
            gradients = torch.autograd.grad(result.square().sum(), [a, b, bias])
            values.append((result, *gradients))
    for old, new in zip(*values):
        same_bits(old, new)


@pytest.mark.addmm
def test_wide_addmm_overlap_guard(monkeypatch):
    module = backend()
    a, b, bias, *_ = operands(1225, 768, 384)
    out = torch.empty(1225, 768, device=a.device)
    monkeypatch.setattr(module, "_WIDE_ENABLED", True)
    with precision(False):
        assert module._wide_plan(a, b, bias, out, 1225, 768, 384, False)
        for overlapping in [a, b, bias]:
            assert not module._wide_plan(a, b, bias, overlapping, 1225, 768, 384, False)
        original = out.normal_().clone()
        with torch.no_grad(), flag_gems.use_gems():
            monkeypatch.setattr(module, "_WIDE_ENABLED", False)
            reference = torch.addmm(original, a, b, alpha=0.25, beta=-0.5)
            monkeypatch.setattr(module, "_WIDE_ENABLED", True)
            torch.addmm(out, a, b, out=out, alpha=0.25, beta=-0.5)
        same_bits(reference, out)


@pytest.mark.addmm
def test_wide_addmm_current_stream(monkeypatch):
    module = backend()
    _, _, _, ah, bh, bias_h = operands(1225, 768, 384)
    stream = torch.cuda.Stream()
    values = []
    with precision(False), torch.cuda.stream(stream), torch.no_grad(), flag_gems.use_gems():
        a = ah.to(flag_gems.device)
        b = bh.contiguous().to(flag_gems.device)
        bias = bias_h.to(flag_gems.device)
        for enabled in [False, True]:
            monkeypatch.setattr(module, "_WIDE_ENABLED", enabled)
            values.append((torch.addmm(bias, a, b) * 0.5).clone())
    torch.cuda.current_stream().wait_stream(stream)
    same_bits(*values)
