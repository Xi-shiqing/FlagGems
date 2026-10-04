"""RNE semantic probe and real FlagGems FP32-matmul entry smoke tests."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.ops.tf32_precision import (
    active_module_restored,
    pop_active_module,
    push_active_module,
    round_tf32_rne,
    tf32_enabled_for_active_module,
)
from flag_gems.ops.mm import mm as flag_mm
from flag_gems.ops.addmm import addmm as flag_addmm
from flag_gems.ops.bmm import bmm as flag_bmm
from protenix.model.tri_attention.flag_gems_public_adapter import (
    flag_gems_public_triangle_attention,
)


@triton.jit
def _round_probe(x_ptr, y_ptr, count: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, offsets < count, other=0.0)
    y = round_tf32_rne(x, True)
    tl.store(y_ptr + offsets, y, offsets < count)


def _host_rne_tf32(bits: int) -> int:
    exponent = bits & 0x7F800000
    if exponent == 0x7F800000:
        return bits
    sign = bits & 0x80000000
    magnitude = bits & 0x7FFFFFFF
    truncated = magnitude & 0xFFFFE000
    remainder = magnitude & 0x1FFF
    retained_lsb = (truncated >> 13) & 1
    if remainder > 0x1000 or (remainder == 0x1000 and retained_lsb):
        truncated += 0x2000
    return sign | (truncated & 0x7FFFFFFF)


def _quantize_tf32_cpu(tensor: torch.Tensor) -> torch.Tensor:
    source = tensor.detach().cpu().contiguous()
    bits = [
        value & 0xFFFFFFFF
        for value in source.reshape(-1).view(torch.int32).tolist()
    ]
    rounded = [_host_rne_tf32(value) for value in bits]
    signed = [value if value < 0x80000000 else value - 0x100000000 for value in rounded]
    return torch.tensor(signed, dtype=torch.int32).view(torch.float32).reshape(source.shape)


def _rne_dot_error(output: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> float:
    expected = _quantize_tf32_cpu(a).double() @ _quantize_tf32_cpu(b).double()
    actual = output.detach().cpu().double()
    return float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))


def _rne_addmm_error(
    output: torch.Tensor, bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> float:
    expected = (
        _quantize_tf32_cpu(a).double() @ _quantize_tf32_cpu(b).double()
    ) + bias.detach().cpu().double()
    actual = output.detach().cpu().double()
    return float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))


def _semantic_probe(device: torch.device) -> dict:
    rng = random.Random(101)
    bits = [
        0x3F801000,  # halfway, lower TF32 value has even mantissa
        0x3F803000,  # halfway, upper TF32 value has even mantissa
        0xBF801000,
        0xBF803000,
        0x00001FFF,
        0x007FFFFF,
        0x7F7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC12345,
        0xFFC12345,
        0x00000000,
        0x80000000,
    ]
    bits.extend(rng.getrandbits(32) for _ in range(8192))
    signed = [value if value < 0x80000000 else value - 0x100000000 for value in bits]
    x = torch.tensor(signed, dtype=torch.int32).view(torch.float32).to(device)
    y = torch.empty_like(x)
    _round_probe[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), 256)
    torch.cuda.synchronize(device)
    actual = [value & 0xFFFFFFFF for value in y.cpu().view(torch.int32).tolist()]
    expected = [_host_rne_tf32(value) for value in bits]
    if actual != expected:
        mismatch = next(i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
        raise AssertionError(
            f"RNE bit mismatch at {mismatch}: got 0x{actual[mismatch]:08x}, "
            f"expected 0x{expected[mismatch]:08x}"
        )
    return {"values": len(bits), "bitwise_match": True}


def _module_restore_probe() -> dict:
    old_patterns = os.environ.get("PROTENIX_TF32_RESTORE_MODULES")
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        os.environ["PROTENIX_TF32_RESTORE_MODULES"] = (
            "diffusion_module.diffusion_transformer.blocks.*.attention.linear_q"
        )
        module = "diffusion_module.diffusion_transformer.blocks.0.attention.linear_q"
        push_active_module(module)
        matched = active_module_restored() and not tf32_enabled_for_active_module()
        pop_active_module(module)
        other = "msa_module.blocks.0.pair_transition.linear_no_bias_a"
        push_active_module(other)
        unmatched = not active_module_restored() and tf32_enabled_for_active_module()
        pop_active_module(other)
        if not (matched and unmatched):
            raise AssertionError({"matched": matched, "unmatched": unmatched})
        return {"matched_restore_uses_strict_fp32": matched, "unmatched_retains_tf32": unmatched}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
        if old_patterns is None:
            os.environ.pop("PROTENIX_TF32_RESTORE_MODULES", None)
        else:
            os.environ["PROTENIX_TF32_RESTORE_MODULES"] = old_patterns


def _entry_probe(device: torch.device) -> dict:
    torch.manual_seed(101)
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        a = torch.randn((32, 48), device=device, dtype=torch.float32)
        b = torch.randn((48, 24), device=device, dtype=torch.float32)
        bias = torch.randn((32, 24), device=device, dtype=torch.float32)
        batched_a = torch.randn((2, 16, 32), device=device, dtype=torch.float32)
        batched_b = torch.randn((2, 32, 24), device=device, dtype=torch.float32)
        with flag_gems.use_gems():
            mm_dispatch = torch.mm(a, b)
            addmm_dispatch = torch.addmm(bias, a, b)
            bmm_dispatch = torch.bmm(batched_a, batched_b)
        mm_direct = flag_mm(a, b)
        addmm_direct = flag_addmm(bias, a, b)
        bmm_direct = flag_bmm(batched_a, batched_b)
        torch.cuda.synchronize(device)
        checks = {
            "mm_dispatch_matches_flaggems": torch.equal(mm_dispatch, mm_direct),
            "addmm_dispatch_matches_flaggems": torch.equal(addmm_dispatch, addmm_direct),
            "bmm_dispatch_matches_flaggems": torch.equal(bmm_dispatch, bmm_direct),
            "finite": all(
                bool(torch.isfinite(value).all())
                for value in (mm_dispatch, addmm_dispatch, bmm_dispatch)
            ),
        }
        if not all(checks.values()):
            raise AssertionError(checks)
        rne_errors = {
            "mm_splitk": _rne_dot_error(mm_dispatch, a, b),
            "addmm_splitk": _rne_addmm_error(addmm_dispatch, bias, a, b),
            "bmm": _rne_dot_error(bmm_dispatch, batched_a, batched_b),
        }

        symmetric = torch.randn((32, 48), device=device, dtype=torch.float32)
        general_a = torch.randn((16, 32), device=device, dtype=torch.float32)
        general_b = torch.randn((32, 513), device=device, dtype=torch.float32)
        with flag_gems.use_gems():
            syrk_dispatch = torch.mm(symmetric, symmetric.t())
            general_dispatch = torch.mm(general_a, general_b)
        syrk_direct = flag_mm(symmetric, symmetric.t())
        general_direct = flag_mm(general_a, general_b)
        checks["mm_syrk_matches_flaggems"] = torch.equal(syrk_dispatch, syrk_direct)
        checks["mm_general_matches_flaggems"] = torch.equal(
            general_dispatch, general_direct
        )
        rne_errors["mm_syrk"] = _rne_dot_error(syrk_dispatch, symmetric, symmetric.t())
        rne_errors["mm_general"] = _rne_dot_error(general_dispatch, general_a, general_b)
        if not all(checks.values()):
            raise AssertionError(checks)
        if any(value > 1e-5 for value in rne_errors.values()):
            raise AssertionError({"RNE relative-L2 mismatch": rne_errors})

        torch.backends.cuda.matmul.allow_tf32 = False
        with flag_gems.use_gems():
            strict_mm = torch.mm(a, b)
            strict_addmm = torch.addmm(bias, a, b)
            strict_bmm = torch.bmm(batched_a, batched_b)
        strict_checks = {
            "mm_dispatch_matches_flaggems": torch.equal(strict_mm, flag_mm(a, b)),
            "addmm_dispatch_matches_flaggems": torch.equal(
                strict_addmm, flag_addmm(bias, a, b)
            ),
            "bmm_dispatch_matches_flaggems": torch.equal(
                strict_bmm, flag_bmm(batched_a, batched_b)
            ),
            "finite": all(
                bool(torch.isfinite(value).all())
                for value in (strict_mm, strict_addmm, strict_bmm)
            ),
        }
        if not all(strict_checks.values()):
            raise AssertionError(strict_checks)

        old_public = os.environ.get("PROTENIX_TRIATTENTION_FLAGGEMS_PUBLIC")
        old_precision = os.environ.get("PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION")
        old_streaming = os.environ.get("PROTENIX_TRIATTENTION_PPU_STREAMING")
        old_small = os.environ.get("PROTENIX_TRIATTENTION_PPU_SMALL")
        os.environ["PROTENIX_TRIATTENTION_FLAGGEMS_PUBLIC"] = "1"
        os.environ["PROTENIX_TRIATTENTION_PPU_STREAMING"] = "0"
        os.environ["PROTENIX_TRIATTENTION_PPU_SMALL"] = "0"
        attention_checks = {}
        try:
            for precision in ("tf32", "qk_tf32", "pv_tf32", "ieee"):
                os.environ["PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION"] = precision
                q = (
                    torch.randn(
                        (1, 2, 8, 1, 16), device=device, dtype=torch.float32
                    )
                    .detach()
                    .requires_grad_()
                )
                k = torch.randn_like(q).detach().requires_grad_()
                v = torch.randn_like(q).detach().requires_grad_()
                bias1 = torch.zeros(
                    (1, 2, 1, 1, 8), device=device, dtype=torch.float32
                )
                bias2 = (
                    torch.zeros(
                        (1, 1, 1, 8, 8), device=device, dtype=torch.float32
                    )
                    .detach()
                    .requires_grad_()
                )
                tri = flag_gems_public_triangle_attention(q, k, v, bias1, bias2)
                tri.backward(torch.randn_like(tri))
                torch.cuda.synchronize(device)
                tri_checks = {
                    "forward_finite": bool(torch.isfinite(tri).all()),
                    "qkv_grad_finite": all(
                        tensor.grad is not None
                        and bool(torch.isfinite(tensor.grad).all())
                        for tensor in (q, k, v)
                    ),
                    "bias2_grad_finite": (
                        bias2.grad is not None and bool(torch.isfinite(bias2.grad).all())
                    ),
                }
                if not all(tri_checks.values()):
                    raise AssertionError({precision: tri_checks})
                attention_checks[precision] = tri_checks
        finally:
            if old_public is None:
                os.environ.pop("PROTENIX_TRIATTENTION_FLAGGEMS_PUBLIC", None)
            else:
                os.environ["PROTENIX_TRIATTENTION_FLAGGEMS_PUBLIC"] = old_public
            if old_precision is None:
                os.environ.pop("PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION", None)
            else:
                os.environ["PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION"] = old_precision
            if old_streaming is None:
                os.environ.pop("PROTENIX_TRIATTENTION_PPU_STREAMING", None)
            else:
                os.environ["PROTENIX_TRIATTENTION_PPU_STREAMING"] = old_streaming
            if old_small is None:
                os.environ.pop("PROTENIX_TRIATTENTION_PPU_SMALL", None)
            else:
                os.environ["PROTENIX_TRIATTENTION_PPU_SMALL"] = old_small
        return {
            "mm_addmm_bmm_tf32": checks,
            "tf32_rne_operand_reference_relative_l2": rne_errors,
            "mm_addmm_bmm_strict_fp32": strict_checks,
            "triangle_adapter_forward_backward": attention_checks,
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda")
    result = {
        "device": torch.cuda.get_device_name(device),
        "semantic": _semantic_probe(device),
        "module_restore": _module_restore_probe(),
        "entries": _entry_probe(device),
    }
    result["passed"] = True
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
