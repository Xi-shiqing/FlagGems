"""Consume packed BMM outputs on the same nondefault stream as producers."""
import importlib

import pytest
import torch

import flag_gems


@pytest.mark.bmm
@pytest.mark.parametrize("tf32", [False, True])
def test_bmm_pack_nondefault_stream(monkeypatch, tf32):
    project = flag_gems.bmm.__globals__["_project_bmm"]
    policy = importlib.import_module(project.__module__)._fp32_pack
    torch.manual_seed(20261110)
    a = torch.randn(32, 128, 129)
    b = torch.randn(32, 129, 127)
    ag, bg = a.to(flag_gems.device), b.to(flag_gems.device)
    torch.cuda.synchronize()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    stream = torch.cuda.Stream()
    try:
        with torch.no_grad(), torch.cuda.stream(stream), flag_gems.use_gems():
            aa = (ag + 1).permute(1, 2, 0).contiguous().permute(2, 0, 1)
            bb = (bg + 1).permute(2, 1, 0).contiguous().permute(2, 1, 0)
            monkeypatch.setattr(policy, "ENABLED", False)
            expected = torch.bmm(aa, bb)
            monkeypatch.setattr(policy, "ENABLED", True)
            actual = torch.bmm(aa, bb)
            consumer = actual * 2
        stream.synchronize()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(consumer, expected * 2)
        if not tf32:
            reference = torch.bmm((a[:2, :16] + 1).double(), (b[:2, :, :16] + 1).double())
            torch.testing.assert_close(actual[:2, :16, :16].cpu().double(), reference, rtol=2e-5, atol=2e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
