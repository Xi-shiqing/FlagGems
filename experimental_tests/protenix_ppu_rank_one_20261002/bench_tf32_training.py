"""Matched-seed GEMM and Triangle Attention backward/update audit."""

import argparse
import contextlib
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch


CASES = (
    ("mm_model_large", "mm", (66049, 128, 512)),
    ("addmm_model_small", "addmm", (257, 1536, 768)),
    ("bmm_model_small", "bmm", (1973, 1, 384, 50)),
    ("triangle_attention_model", "triattention", (1, 257, 257, 4, 32)),
)
LR = 1e-3


def _array(rng, shape, scale):
    return rng.integers(-32768, 32768, size=shape, dtype=np.int32).astype(
        np.float32
    ) * scale


def _inputs(kind, shape, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    if kind == "triattention":
        batch, n, s, heads, dim = shape
        values = {
            "q": _array(rng, shape, 2.0**-16),
            "k": _array(rng, shape, 2.0**-16),
            "v": _array(rng, shape, 2.0**-16),
            "bias1": _array(rng, (batch, n, 1, 1, s), 2.0**-16),
            "bias2": _array(rng, (batch, 1, heads, s, s), 2.0**-16),
            "upstream": _array(rng, shape, 2.0**-20),
        }
    elif kind in {"mm", "addmm"}:
        m, k, n = shape
        weight = _array(rng, (n, k), 2.0**-18)
        a = _array(rng, (m, k), 2.0**-18)
        upstream = _array(rng, (m, n), 2.0**-20)
        bias = _array(rng, (n,), 2.0**-18) if kind == "addmm" else None
        values = {"a": a, "weight": weight, "upstream": upstream}
        if bias is not None:
            values["bias"] = bias
    else:
        batch, m, k, n = shape
        values = {
            "a": _array(rng, (batch, m, k), 2.0**-18),
            "weight": _array(rng, (batch, k, n), 2.0**-18),
            "upstream": _array(rng, (batch, m, n), 2.0**-20),
        }
    return values


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _relative_l2(actual, reference):
    a = actual.detach().double().cpu()
    b = reference.detach().double().cpu()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-300))


def _triangle_output(q, k, v, bias1, bias2, backend, enabled):
    if backend == "ppu":
        os.environ["PROTENIX_TRIATTENTION_FLAGGEMS_PUBLIC"] = "1"
        os.environ["PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION"] = (
            "tf32" if enabled else "ieee"
        )
        os.environ["PROTENIX_TRIATTENTION_PPU_STREAMING"] = "0"
        os.environ["PROTENIX_TRIATTENTION_PPU_SMALL"] = "0"
        from protenix.model.tri_attention.flag_gems_public_adapter import (
            flag_gems_public_triangle_attention,
        )

        return flag_gems_public_triangle_attention(q, k, v, bias1, bias2)
    qh = q.permute(0, 1, 3, 2, 4)
    kh = k.permute(0, 1, 3, 4, 2)
    vh = v.permute(0, 1, 3, 2, 4)
    scores = torch.matmul(qh, kh) + bias1 + bias2
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, vh).permute(0, 1, 3, 2, 4)


def _evaluate_triangle_attention(arrays, device, backend, enabled):
    q, k, v = [
        torch.from_numpy(arrays[key]).to(device).requires_grad_()
        for key in ("q", "k", "v")
    ]
    bias1 = torch.from_numpy(arrays["bias1"]).to(device)
    bias2 = torch.from_numpy(arrays["bias2"]).to(device).requires_grad_()
    upstream = torch.from_numpy(arrays["upstream"]).to(device)
    with torch.enable_grad():
        output = _triangle_output(q, k, v, bias1, bias2, backend, enabled)
        grads = torch.autograd.grad(output, (q, k, v, bias2), upstream)
    params = (q, k, v, bias2)
    keys = ("query", "key", "value", "bias2")
    result = {"output": output.detach().cpu()}
    for key, param, grad in zip(keys, params, grads):
        updated = param.detach() - LR * grad.detach()
        result[f"grad_{key}"] = grad.detach().cpu()
        result[f"updated_{key}"] = updated.cpu()
        result[f"{key}_update"] = (updated - param.detach()).cpu()
    return result


def _evaluate(kind, arrays, device, backend, enabled):
    if kind == "triattention":
        return _evaluate_triangle_attention(arrays, device, backend, enabled)
    a = torch.from_numpy(arrays["a"]).to(device).requires_grad_()
    weight = torch.from_numpy(arrays["weight"]).to(device).requires_grad_()
    upstream = torch.from_numpy(arrays["upstream"]).to(device)
    bias = (
        torch.from_numpy(arrays["bias"]).to(device).requires_grad_()
        if "bias" in arrays
        else None
    )
    with torch.enable_grad():
        if kind == "mm":
            output = torch.mm(a, weight.t())
        elif kind == "addmm":
            output = torch.addmm(bias, a, weight.t())
        else:
            output = torch.bmm(a, weight)
        params = (a, weight, bias) if bias is not None else (a, weight)
        grads = torch.autograd.grad(output, params, upstream)
    updated_weight = weight.detach() - LR * grads[1].detach()
    result = {
        "output": output.detach().cpu(),
        "grad_activation": grads[0].detach().cpu(),
        "grad_weight": grads[1].detach().cpu(),
        "updated_weight": updated_weight.cpu(),
        "weight_update": (updated_weight - weight.detach()).cpu(),
    }
    if bias is not None:
        result["grad_bias"] = grads[2].detach().cpu()
        result["updated_bias"] = bias.detach().cpu() - LR * grads[2].detach().cpu()
        result["bias_update"] = result["updated_bias"] - bias.detach().cpu()
    return result


def _train_step(kind, a, weight, upstream, bias):
    with torch.enable_grad():
        if kind == "mm":
            output = torch.mm(a, weight.t())
        elif kind == "addmm":
            output = torch.addmm(bias, a, weight.t())
        else:
            output = torch.bmm(a, weight)
        params = (a, weight, bias) if bias is not None else (a, weight)
        grads = torch.autograd.grad(output, params, upstream)
    with torch.no_grad():
        weight.add_(grads[1], alpha=-LR)
        if bias is not None:
            bias.add_(grads[2], alpha=-LR)


def _train_triangle_step(q, k, v, bias1, bias2, upstream, backend, enabled):
    with torch.enable_grad():
        output = _triangle_output(q, k, v, bias1, bias2, backend, enabled)
        grads = torch.autograd.grad(output, (q, k, v, bias2), upstream)
    with torch.no_grad():
        for param, grad in zip((q, k, v, bias2), grads):
            param.add_(grad, alpha=-LR)


def _time_triangle_attention(arrays, device, backend, enabled, context_factory, repeats):
    q, k, v = [
        torch.from_numpy(arrays[key]).to(device).requires_grad_()
        for key in ("q", "k", "v")
    ]
    bias1 = torch.from_numpy(arrays["bias1"]).to(device)
    bias2 = torch.from_numpy(arrays["bias2"]).to(device).requires_grad_()
    upstream = torch.from_numpy(arrays["upstream"]).to(device)
    times = []
    with context_factory():
        for _ in range(2):
            _train_triangle_step(q, k, v, bias1, bias2, upstream, backend, enabled)
        torch.cuda.synchronize(device)
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            before = time.perf_counter()
            start.record()
            _train_triangle_step(q, k, v, bias1, bias2, upstream, backend, enabled)
            end.record()
            torch.cuda.synchronize(device)
            times.append({
                "wall_ms": (time.perf_counter() - before) * 1000,
                "event_ms": start.elapsed_time(end),
            })
    return {
        "repeats": times,
        "median_wall_ms": statistics.median(row["wall_ms"] for row in times),
        "median_event_ms": statistics.median(row["event_ms"] for row in times),
    }


def _time(kind, arrays, device, backend, enabled, context_factory, repeats=5):
    if kind == "triattention":
        return _time_triangle_attention(
            arrays, device, backend, enabled, context_factory, repeats
        )
    a = torch.from_numpy(arrays["a"]).to(device).requires_grad_()
    weight0 = torch.from_numpy(arrays["weight"]).to(device)
    upstream = torch.from_numpy(arrays["upstream"]).to(device)
    bias0 = torch.from_numpy(arrays["bias"]).to(device) if "bias" in arrays else None
    weight = weight0.clone().requires_grad_()
    bias = bias0.clone().requires_grad_() if bias0 is not None else None
    times = []
    with context_factory():
        for _ in range(2):
            _train_step(kind, a, weight, upstream, bias)
        torch.cuda.synchronize(device)
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            before = time.perf_counter()
            start.record()
            _train_step(kind, a, weight, upstream, bias)
            end.record()
            torch.cuda.synchronize(device)
            times.append({
                "wall_ms": (time.perf_counter() - before) * 1000,
                "event_ms": start.elapsed_time(end),
            })
    return {
        "repeats": times,
        "median_wall_ms": statistics.median(row["wall_ms"] for row in times),
        "median_event_ms": statistics.median(row["event_ms"] for row in times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ppu", "h100"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--case", choices=[case[0] for case in CASES])
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    if args.backend == "ppu":
        import flag_gems

        context_factory = lambda: flag_gems.use_gems(exclude=[])
    else:
        context_factory = contextlib.nullcontext

    report = {
        "backend": args.backend,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "input_generator": "NumPy PCG64, fixed seed, dyadic FP32 values",
        "lr": LR,
        "results": [],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, (name, kind, shape) in enumerate(CASES):
        if args.case and name != args.case:
            continue
        seed = args.seed + index
        arrays = _inputs(kind, shape, seed)
        hashes = {key: _sha(value) for key, value in arrays.items()}
        row = {
            "case": name,
            "op": kind,
            "shape": list(shape),
            "seed": seed,
            "input_sha256": hashes,
        }
        try:
            modes = {}
            for mode, enabled in (("fp32", False), ("tf32", True)):
                torch.backends.cuda.matmul.allow_tf32 = enabled
                torch.set_float32_matmul_precision("high" if enabled else "highest")
                with context_factory():
                    modes[mode] = _evaluate(kind, arrays, device, args.backend, enabled)
                # _time owns its own backend context; keeping it outside the
                # evaluation context avoids nested FlagGems registrar teardown.
                modes[mode]["timing"] = _time(
                    kind, arrays, device, args.backend, enabled, context_factory
                )
            errors = {}
            for key in modes["fp32"]:
                if key == "timing":
                    continue
                lhs, rhs = modes["tf32"][key], modes["fp32"][key]
                errors[key] = {
                    "relative_l2_to_fp32": _relative_l2(lhs, rhs),
                    "max_abs_to_fp32": float((lhs - rhs).abs().max()),
                    "finite": bool(torch.isfinite(lhs).all()),
                }
            row.update({
                "tf32_vs_fp32": errors,
                "output_sha256": {
                    mode: {
                        key: _sha(value.detach().cpu().numpy())
                        for key, value in modes[mode].items() if key != "timing"
                    }
                    for mode in ("fp32", "tf32")
                },
                "timing": {
                    mode: modes[mode]["timing"] for mode in ("fp32", "tf32")
                },
                "tf32_over_fp32_wall_speedup": (
                    modes["fp32"]["timing"]["median_wall_ms"]
                    / modes["tf32"]["timing"]["median_wall_ms"]
                ),
            })
        except Exception as exc:
            row["error"] = repr(exc)
        report["results"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row, separators=(",", ":")), flush=True)
        torch.cuda.empty_cache()
    print(json.dumps({"complete": True, "out": str(output)}))
    if any("error" in row for row in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
