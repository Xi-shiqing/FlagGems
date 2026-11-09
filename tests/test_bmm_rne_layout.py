"""Check layout conversion against independent CPU integer RNE, bit for bit."""
import importlib
import math

import pytest
import torch

import flag_gems
from flag_gems.runtime import device as runtime_device


@pytest.mark.bmm
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU operand layout")
@pytest.mark.parametrize("order", ["contiguous", "batch_fast", "transposed_matrix"])
@pytest.mark.parametrize("shape", [(33, 65, 127), (64, 127, 255)])
def test_rne_dense_layout_against_cpu_bits(order, shape):
    project = flag_gems.bmm.__globals__["_project_bmm"]
    prefix = project.__module__.rsplit(".", 1)[0]
    module = importlib.import_module(prefix + ".bmm_rne_layout")
    generator = torch.Generator(device="cpu").manual_seed(20261107)
    cpu = torch.randn(shape, generator=generator)
    special = torch.tensor(
        [0, -2147483648, 0x3F801000, 0x3F803000, 0x7F800000,
         -8388608, 0x7FC00001, -4194303, 0x00001000, 0x00003000],
        dtype=torch.int32,
    )
    cpu.view(torch.int32).reshape(-1)[:special.numel()] = special
    x = cpu.to(flag_gems.device)
    if order == "batch_fast":
        x = x.permute(1, 2, 0).contiguous().permute(2, 0, 1)
    elif order == "transposed_matrix":
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    # A nonzero storage offset must not be mistaken for the storage base.
    storage = torch.empty(math.prod(shape) + 7, device=x.device, dtype=x.dtype)
    shifted = storage.as_strided(shape, x.stride(), 7)
    shifted.copy_(x)
    actual = module._round_dense(shifted).cpu().view(torch.int32)
    bits = cpu.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    finite = (bits & 0x7F800000) != 0x7F800000
    rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & 0xFFFFE000
    expected = torch.where(finite, rounded, bits).to(torch.int32)
    assert torch.equal(actual, expected)


@pytest.mark.bmm
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU operand layout")
@pytest.mark.parametrize("transpose_out", [False, True])
def test_rne_registered_bmm_out_matches_rollback(transpose_out, monkeypatch):
    generator = torch.Generator(device="cpu").manual_seed(20261107)
    batch, rows, reduction, cols = 65, 511, 257, 509
    a = torch.randn((batch, rows, reduction), generator=generator).to(flag_gems.device)
    b = torch.randn((batch, reduction, cols), generator=generator).to(flag_gems.device)
    a = a.permute(1, 2, 0).contiguous().permute(2, 0, 1)
    b = b.permute(2, 1, 0).contiguous().permute(2, 1, 0)
    shape = (batch, cols, rows) if transpose_out else (batch, rows, cols)
    out = torch.empty(shape, device=a.device, dtype=a.dtype)
    if transpose_out:
        out = out.transpose(1, 2)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with flag_gems.use_gems():
            monkeypatch.setenv("FLAG_GEMS_PPU_BMM_LAYOUT_RNE", "0")
            expected = torch.bmm(a, b)
            monkeypatch.setenv("FLAG_GEMS_PPU_BMM_LAYOUT_RNE", "1")
            actual = torch.ops.aten.bmm.out(a, b, out=out)
        assert actual.data_ptr() == out.data_ptr()
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
