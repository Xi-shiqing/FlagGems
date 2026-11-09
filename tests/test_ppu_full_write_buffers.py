"""Poison final outputs and partial sums, including edge tiles and empty K."""

import importlib

import pytest
import torch
import torch.nn.functional as F

import flag_gems
from flag_gems.runtime import device as runtime_device
from flag_gems.utils import native_output_allocation as allocator
from test_full_write_allocation import checked_allocator

pytestmark = pytest.mark.skipif(
    runtime_device.vendor_name != "thead" or not allocator.ENABLED,
    reason="requires PPU and the optional native allocator",
)

mm = importlib.import_module("flag_gems.runtime.backend._thead.ops.mm_fp32")
addmm = importlib.import_module("flag_gems.runtime.backend._thead.ops.addmm_fp32")
sdpa = importlib.import_module("flag_gems.runtime.backend._thead.ops.sdpa_math")


@pytest.fixture(params=[False, True], ids=["fp32", "tf32"])
def precision(request):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = request.param
    try:
        yield request.param
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def pair(monkeypatch, operation):
    native = allocator.empty_for_overwrite
    with torch.no_grad():
        monkeypatch.setattr(allocator, "empty_for_overwrite", torch.empty)
        control = operation()
        monkeypatch.setattr(allocator, "empty_for_overwrite", native)
        candidate = operation()
    return control, candidate


def inputs(m, n, k, layout):
    rng = torch.Generator().manual_seed(20261109)
    a = torch.randn((m, k), generator=rng)
    b = torch.randn((k, n), generator=rng)
    a, b = a.to(flag_gems.device), b.to(flag_gems.device)
    if layout == "transpose":
        a = a.T.contiguous().T
        b = b.T.contiguous().T
    return a, b


def cpu_product(a, b, tf32):
    a, b = a.cpu().contiguous(), b.cpu().contiguous()
    if tf32:
        # Model TF32 input rounding independently on CPU, then accumulate in
        # float64. The remaining check measures accumulation, not quantization.
        def rne(x):
            bits = x.view(torch.int32)
            return ((bits + 0xFFF + ((bits >> 13) & 1)) & -8192).view(torch.float32)
        a, b = rne(a), rne(b)
    return a.double() @ b.double()


@pytest.mark.mm
@pytest.mark.parametrize("shape", [(7, 19, 0), (7, 19, 13), (65, 67, 129), (17, 513, 33), (3, 31, 4097)])
@pytest.mark.parametrize("layout", ["contiguous", "transpose"])
def test_mm_buffers(monkeypatch, checked_allocator, precision, shape, layout, record_property):
    a, b = inputs(*shape, layout)
    # Exercise both generic and split-K producers, including uneven chunks.
    monkeypatch.setattr(mm, "_DIRECT_ENABLED", False)
    old, new = pair(monkeypatch, lambda: mm.mm(a, b))
    assert torch.equal(old, new)
    exact = cpu_product(a, b, False).float()
    reference = cpu_product(a, b, precision).float()
    record_property("strict_fp32_max_abs_error", (new.cpu() - exact).abs().max().item())
    record_property("same_precision_accumulation_max_abs_error", (new.cpu() - reference).abs().max().item())
    record_property("fixed_tolerance_diagnostic_pass", bool(torch.all(
        (new.cpu() - reference).abs() <= 2e-4 + 2e-5 * reference.abs()
    )))
    # Use the existing MM suite's reduction-length-aware CPU criterion.
    # Allocation changes have the stronger, separate bitwise equality check.
    flag_gems.testing.assert_close(new.cpu(), reference, torch.float32, reduce_dim=shape[2])
    if shape[2] > 0:
        assert checked_allocator
    assert all(torch.isfinite(t).all() for t in checked_allocator)


@pytest.mark.mm
@pytest.mark.parametrize("shape", [(7, 0), (65, 67), (129, 33)])
def test_symmetric_output(monkeypatch, checked_allocator, precision, shape):
    a, _ = inputs(shape[0], 1, shape[1], "transpose")
    old, new = pair(monkeypatch, lambda: mm.mm(a, a.T))
    assert torch.equal(old, new)
    torch.testing.assert_close(new.cpu(), (a.cpu().double() @ a.cpu().double().T).float(),
                               atol=0.02 if precision else 2e-4,
                               rtol=0.003 if precision else 2e-5)
    if shape[1] > 0:
        assert checked_allocator
    assert all(torch.isfinite(t).all() for t in checked_allocator)


@pytest.mark.addmm
@pytest.mark.parametrize("shape", [(7, 19, 0), (65, 67, 129), (17, 513, 33)])
@pytest.mark.parametrize("given_out", [False, True])
@pytest.mark.parametrize("bias_kind", ["row", "matrix"])
def test_addmm_buffers(monkeypatch, checked_allocator, precision, shape, given_out, bias_kind):
    a, b = inputs(*shape, "transpose")
    m, n, _ = shape
    bias = torch.full((n,) if bias_kind == "row" else (m, n), 0.25, device=flag_gems.device)
    out = torch.full((m, n), float("nan"), device=flag_gems.device)
    def operation():
        if given_out:
            return addmm.addmm_out(bias, a, b, beta=0.5, alpha=-0.75, out=out).clone()
        return addmm.addmm(bias, a, b, beta=0.5, alpha=-0.75)
    old, new = pair(monkeypatch, operation)
    assert torch.equal(old, new)
    reference = (a.cpu().double() @ b.cpu().double()) * -0.75 + bias.cpu().double() * 0.5
    torch.testing.assert_close(new.cpu(), reference.float(), atol=0.02 if precision else 1e-4,
                               rtol=0.003 if precision else 2e-5)
    if not given_out or shape[1] <= 512:
        assert checked_allocator
    assert all(torch.isfinite(t).all() for t in checked_allocator)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.parametrize("mask_kind", ["none", "float", "masked_row"])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice"])
def test_attention_buffers(monkeypatch, checked_allocator, precision, mask_kind, layout):
    rng = torch.Generator().manual_seed(8040)
    def tensor(shape):
        if layout == "transpose":
            return torch.randn((*shape[:-2], shape[-1], shape[-2]), generator=rng).to(flag_gems.device).transpose(-1, -2)
        if layout == "slice":
            return torch.randn((*shape[:-1], shape[-1] * 2), generator=rng).to(flag_gems.device)[..., ::2]
        return torch.randn(shape, generator=rng).to(flag_gems.device)
    q, k, v = tensor((1, 3, 17, 32)), tensor((1, 3, 63, 32)), tensor((1, 3, 63, 19))
    mask = None if mask_kind == "none" else torch.randn((17, 63), generator=rng).to(flag_gems.device)
    if mask_kind == "masked_row":
        mask[0] = -float("inf")
    old, new = pair(monkeypatch, lambda: sdpa._ordered_forward(q, k, v, mask))
    assert all(torch.equal(a, b) for a, b in zip(old, new))
    reference = F.scaled_dot_product_attention(q.cpu().double(), k.cpu().double(), v.cpu().double(),
                                               attn_mask=None if mask is None else mask.cpu().double())
    torch.testing.assert_close(new[0].cpu(), reference.float(), atol=0.008 if precision else 1e-5,
                               rtol=0.008 if precision else 1e-5)
    assert len(checked_allocator) >= 4
    assert all(torch.isfinite(t).all() for t in checked_allocator)


@pytest.mark.mm
@pytest.mark.addmm
def test_grad_enabled_keeps_original_allocation(monkeypatch, checked_allocator):
    def forbidden(*args, **kwargs):
        raise AssertionError("grad-enabled call used the inference allocator")
    monkeypatch.setattr(allocator, "_empty", forbidden)
    a, b = inputs(7, 19, 13, "contiguous")
    with torch.enable_grad():
        assert torch.isfinite(mm.mm(a, b)).all()
        assert torch.isfinite(addmm.addmm(torch.ones((19,), device=flag_gems.device), a, b)).all()
    assert not checked_allocator


@pytest.mark.mm
@pytest.mark.parametrize("shape", [(0, 19, 13), (7, 0, 13), (7, 19, 0)])
@pytest.mark.parametrize("layout", ["transpose", "slice"])
def test_empty_matrix_and_given_output(checked_allocator, precision, shape, layout):
    a, b = inputs(*shape, "contiguous")
    m, n, _ = shape
    backing = torch.full((m, n * 2), 53.0, device=flag_gems.device)
    out = (torch.full((n, m), float("nan"), device=flag_gems.device).T
           if layout == "transpose" else backing[:, ::2])
    with torch.no_grad():
        actual = mm.mm(a, b)
        returned = mm.mm_out(a, b, out=out)
    assert returned is out
    expected = torch.zeros((m, n))
    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(out.cpu(), expected)
    assert torch.equal(backing[:, 1::2].cpu(), torch.full((m, n), 53.0))


@pytest.mark.mm
def test_packed_rhs_overwrite(monkeypatch, checked_allocator):
    a, b = inputs(65, 67, 129, "transpose")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    monkeypatch.setattr(mm, "_tf32_rne_rhs_prepack_enabled", lambda: True)
    monkeypatch.setattr(mm, "_tf32_rne_rhs_cache_capacity", lambda: 0)
    out = torch.empty((65, 67), device=flag_gems.device)
    def operation():
        mm._TF32_RNE_RHS_CACHE.clear()
        return mm._direct_mm(a, b, out, 65, 67, 129).clone()
    try:
        old, new = pair(monkeypatch, operation)
        assert torch.equal(old, new)
        assert checked_allocator
        assert all(torch.isfinite(t).all() for t in checked_allocator)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
