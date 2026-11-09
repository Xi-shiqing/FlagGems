"""Layout-only copies retain the existing dot order and full output bits."""
import importlib

import pytest
import torch

import flag_gems


def backend():
    return importlib.import_module(flag_gems.bmm.__globals__["_project_bmm"].__module__)


def interleaved(x, reverse=False):
    order = (2, 1, 0) if reverse else (1, 2, 0)
    inverse = tuple(order.index(i) for i in range(3))
    return x.permute(order).contiguous().permute(inverse)


@pytest.mark.bmm
@pytest.mark.parametrize("layout", ["both", "reverse", "a_only", "b_only"])
@pytest.mark.parametrize("tf32", [False, True])
def test_public_bmm_fp32_pack(monkeypatch, layout, tf32):
    module = backend()
    policy = module._fp32_pack
    torch.manual_seed(20261110)
    a = torch.randn(32, 128, 129)
    b = torch.randn(32, 129, 127)
    ag = a.to(flag_gems.device)
    bg = b.to(flag_gems.device)
    if layout != "b_only":
        ag = interleaved(ag, layout == "reverse")
    if layout != "a_only":
        bg = interleaved(bg, layout != "reverse")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        monkeypatch.setattr(policy, "ENABLED", False)
        with torch.no_grad():
            expected = flag_gems.bmm(ag, bg)
        monkeypatch.setattr(policy, "ENABLED", True)
        with torch.no_grad():
            packed = policy.prepare(ag, bg)
            assert packed is not None
            assert packed[0].is_contiguous() and packed[1].is_contiguous()
            assert torch.equal(packed[0], ag) and torch.equal(packed[1], bg)
            actual = flag_gems.bmm(ag, bg)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        if not tf32:
            reference = torch.bmm(a[:2, :16].double(), b[:2, :, :16].double())
            torch.testing.assert_close(actual[:2, :16, :16].cpu().double(), reference, rtol=2e-5, atol=2e-5)
            output = torch.empty_like(actual).transpose(1, 2).contiguous().transpose(1, 2)
            returned = flag_gems.bmm_out(ag, bg, output)
            assert returned is output
            assert torch.equal(returned.contiguous().view(torch.uint8), expected.view(torch.uint8))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.bmm
def test_bmm_pack_guards_and_oom(monkeypatch):
    policy = backend()._fp32_pack
    monkeypatch.setattr(policy, "ENABLED", True)
    a = interleaved(torch.ones(32, 128, 129, device=flag_gems.device))
    b = interleaved(torch.ones(32, 129, 127, device=flag_gems.device))
    assert policy.prepare(a, b, a) is None
    assert policy.prepare(a.cpu(), b.cpu()) is None
    assert policy.prepare(a.half(), b.half()) is None
    assert policy.prepare(a[:16], b[:16]) is None
    assert policy.prepare(a[:, ::2], b) is None
    assert not policy.eligible_layout((256, 2048, 2048), (1, 524288, 256), (256, 2048, 2048), (1, 256, 524288))
    assert not policy.eligible_layout((32, 1, 129), (1, 4128, 32), (32, 129, 127), (1, 4064, 32))
    assert not policy.eligible_layout((32, 128, 129), (16512, 129, 1), (32, 129, 127), (16383, 127, 1))

    def oom(*args, **kwargs):
        raise torch.OutOfMemoryError("test allocation exhaustion")

    monkeypatch.setattr(policy, "clone", oom)
    assert policy.prepare(a, b) is None


@pytest.mark.bmm
def test_registered_bmm_pack_backward(monkeypatch):
    module = backend()
    policy = module._fp32_pack
    torch.manual_seed(20261110)
    a = torch.randn(32, 128, 129, device=flag_gems.device)
    b = torch.randn(32, 129, 127, device=flag_gems.device)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    values = []
    try:
        for enabled in [False, True]:
            monkeypatch.setattr(policy, "ENABLED", enabled)
            aa = interleaved(a).detach().requires_grad_()
            bb = interleaved(b, True).detach().requires_grad_()
            with flag_gems.use_gems():
                output = torch.bmm(aa, bb)
                gradients = torch.autograd.grad(output.square().sum(), [aa, bb])
            values.append((output, *gradients))
        for old, new in zip(*values):
            assert torch.equal(old.contiguous().view(torch.uint8), new.contiguous().view(torch.uint8))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
