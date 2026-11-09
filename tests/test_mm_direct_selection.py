"""Keep fused inference MM when ordered MM would add RHS preparation."""
import importlib
import sys

import pytest
import torch

import flag_gems
from flag_gems.runtime import device as runtime_device


def _observe(monkeypatch):
    backend = sys.modules[flag_gems.mm.__globals__["_project_mm"].__module__]
    ordered = importlib.import_module(backend.__name__.rsplit(".", 1)[0] + ".mm_ordered_direct")
    counts = {"ordered": 0, "existing": 0, "prepared": 0}
    direct, attempt, prepare = backend._direct_mm, ordered.try_mm, ordered._prepare_rhs

    def existing(*args, **kwargs):
        counts["existing"] += 1
        return direct(*args, **kwargs)

    def candidate(*args, **kwargs):
        result = attempt(*args, **kwargs)
        counts["ordered"] += int(result is not None)
        return result

    def prepared(b, tf32):
        result = prepare(b, tf32)
        counts["prepared"] += int(result is not b)
        return result

    monkeypatch.setattr(backend, "_PPU_PACK_RHS_AVAILABLE", True)
    monkeypatch.setattr(backend, "_DIRECT_ENABLED", True)
    monkeypatch.setattr(backend, "_direct_mm", existing)
    monkeypatch.setattr(ordered, "try_mm", candidate)
    monkeypatch.setattr(ordered, "_prepare_rhs", prepared)
    monkeypatch.setenv("FLAG_GEMS_PPU_ORDERED_DIRECT_MM", "1")
    monkeypatch.setenv("FLAG_GEMS_PPU_H100_SPLITK", "1")
    monkeypatch.setenv("FLAG_GEMS_PPU_DIRECT_TF32", "1")
    monkeypatch.setenv("FLAG_GEMS_PPU_TF32_RNE_PACK_RHS", "0")
    return counts


def _inputs(m=17001, transposed=True):
    generator = torch.Generator().manual_seed(20261108)
    a = torch.randn((m, 128), generator=generator)
    b = torch.randn((256, 128) if transposed else (128, 256), generator=generator)
    return a, b.t() if transposed else b


@pytest.mark.mm
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU MM selection")
@pytest.mark.parametrize("tf32", [False, True])
@pytest.mark.parametrize("api", ["mm", "out", "out_transposed"])
def test_registered_mm_avoids_redundant_pack(tf32, api, monkeypatch):
    counts = _observe(monkeypatch)
    cpu_a, cpu_b = _inputs()
    a, b = cpu_a.to(flag_gems.device), cpu_b.to(flag_gems.device)
    previous = torch.backends.cuda.matmul.allow_tf32
    outputs = []
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        for enabled in ["0", "1"]:
            monkeypatch.setenv("FLAG_GEMS_PPU_AVOID_REDUNDANT_MM_PACK", enabled)
            counts.update(dict.fromkeys(counts, 0))
            out = torch.empty((256, 17001) if api == "out_transposed" else (17001, 256), device=a.device)
            if api == "out_transposed":
                out = out.t()
            with torch.no_grad(), flag_gems.use_gems():
                result = torch.mm(a, b) if api == "mm" else torch.ops.aten.mm.out(a, b, out=out)
            if api != "mm":
                assert result.data_ptr() == out.data_ptr()
            outputs.append(result.cpu())
            assert counts == ({"ordered": 1, "existing": 0, "prepared": 1} if enabled == "0"
                              else {"ordered": 0, "existing": 1, "prepared": 0})
        assert torch.equal(outputs[0].view(torch.int32), outputs[1].view(torch.int32))
        reference = cpu_a[:16].double() @ cpu_b[:, :16].double()
        error = (outputs[1][:16, :16].double() - reference).norm() / reference.norm()
        assert float(error) < (0.002 if tf32 else 2e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.mm
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU MM selection")
@pytest.mark.parametrize("tf32,case", [(p, c) for p in [False, True]
                                     for c in ["large", "contiguous", "gradient"]] + [(True, "explicit_pack")])
def test_registered_mm_retains_ordered_path(tf32, case, monkeypatch):
    counts = _observe(monkeypatch)
    cpu_a, cpu_b = _inputs(m=131073 if case == "large" else 17001,
                           transposed=case != "contiguous")
    previous = torch.backends.cuda.matmul.allow_tf32
    outputs = []
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        if case == "explicit_pack":
            monkeypatch.setenv("FLAG_GEMS_PPU_TF32_RNE_PACK_RHS", "1")
        for enabled in ["0", "1"]:
            monkeypatch.setenv("FLAG_GEMS_PPU_AVOID_REDUNDANT_MM_PACK", enabled)
            counts.update(dict.fromkeys(counts, 0))
            a = cpu_a.to(flag_gems.device).requires_grad_(case == "gradient")
            b = cpu_b.to(flag_gems.device).requires_grad_(case == "gradient")
            with torch.set_grad_enabled(case == "gradient"), flag_gems.use_gems():
                result = torch.mm(a, b)
                assert counts["ordered"] == 1 and counts["existing"] == 0
                if case == "gradient":
                    dy = torch.ones_like(result)
                    result.backward(dy)
            outputs.append(tuple(t.detach().cpu() for t in
                                 ((result, a.grad, b.grad) if case == "gradient" else (result,))))
        for actual, expected in zip(outputs[1], outputs[0]):
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
