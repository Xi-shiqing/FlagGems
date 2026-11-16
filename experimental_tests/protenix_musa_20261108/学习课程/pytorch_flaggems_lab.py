"""Practice operator registration, correctness checks and synchronized timing."""
import argparse
import inspect
import json
import statistics
import time
from contextlib import nullcontext

import torch


def cpu_dispatch():
    library = torch.library.Library("protenix_lesson", "DEF")
    library.define("shift_relu(Tensor x, float shift) -> Tensor")
    calls = 0

    def shift_relu(x, shift):
        nonlocal calls
        calls += 1
        return torch.relu(x + shift)

    library.impl("shift_relu", shift_relu, "CPU")
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        result = torch.ops.protenix_lesson.shift_relu(torch.tensor([-2.0, 0.0, 3.0]), 1.0)
    torch.testing.assert_close(result, torch.tensor([0.0, 1.0, 4.0]), atol=0, rtol=0)
    assert calls == 1
    return {"calls": calls, "output": result.tolist(),
            "observed_operator_names": [row.key for row in profiler.key_averages()]}


def benchmark(device, repeats, warmup, gems_log):
    backend = None if device == "cpu" else getattr(torch, device)
    if backend is None:
        synchronize = lambda: None
        context = nullcontext()
        source = None
    else:
        if not backend.is_available():
            raise RuntimeError(f"{device} unavailable")
        free, total = backend.mem_get_info()
        if total - free >= 1024**3:
            raise RuntimeError("Select an idle device explicitly before running the lab")
        import flag_gems
        synchronize = backend.synchronize
        context = flag_gems.use_gems(include=["bmm"])
        source = inspect.getfile(flag_gems)

    generator = torch.Generator().manual_seed(7)
    left = torch.randint(-2, 3, (2, 33, 65), generator=generator).float()
    right = torch.randint(-2, 3, (2, 31, 65), generator=generator).float().transpose(-1, -2)
    expected = torch.bmm(left.double(), right.double()).float()
    left, right = left.to(device), right.to(device)
    same_values_contiguous = right.contiguous()
    cases = {"transpose_view": right, "contiguous": same_values_contiguous}
    samples = {name: [] for name in cases}
    events = {name: [] for name in cases}
    with torch.inference_mode(), context:
        for value in cases.values():
            actual = torch.bmm(left, value)
            torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
            for _ in range(warmup):
                torch.bmm(left, value)
        synchronize()
        for index in range(repeats):
            order = list(cases)
            if index % 2:
                order.reverse()
            for name in order:
                start_event = backend.Event(enable_timing=True) if backend else None
                end_event = backend.Event(enable_timing=True) if backend else None
                synchronize()
                started = time.perf_counter()
                if start_event:
                    start_event.record()
                torch.bmm(left, cases[name])
                if end_event:
                    end_event.record()
                synchronize()
                samples[name].append((time.perf_counter() - started) * 1000)
                if start_event:
                    events[name].append(start_event.elapsed_time(end_event))
    if gems_log:
        if backend is None:
            raise ValueError("--gems-log requires a GPU device")
        with torch.inference_mode(), flag_gems.use_gems(
            include=["bmm"], record=True, path=gems_log
        ):
            torch.bmm(left, right)
            synchronize()
    return {
        "torch": torch.__version__, "device": device, "flaggems_file": source,
        "repeats": repeats, "warmup": warmup, "reference": "CPU FP64, exact integer fixture",
        "cases": {name: {"stride": list(value.stride()), "wall_ms": samples[name],
                         "wall_median_ms": statistics.median(samples[name]),
                         "stream_event_ms": events[name]}
                  for name, value in cases.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "musa"], default="cpu")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gems-log")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup non-negative")
    if args.device == "musa":
        import torch_musa  # Registers the backend in this process.
    print(json.dumps({"dispatch": cpu_dispatch(),
                      "benchmark": benchmark(args.device, args.repeats, args.warmup, args.gems_log)},
                     indent=2))


if __name__ == "__main__":
    main()
