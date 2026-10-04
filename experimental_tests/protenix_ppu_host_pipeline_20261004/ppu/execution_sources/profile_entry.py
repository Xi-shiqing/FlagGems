"""Profile current modes after warmup; raw traces stay on their original server."""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import runpy
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile-entry', type=Path, required=True)
    parser.add_argument('--profile-root', type=Path, required=True)
    parser.add_argument('--allocation-sites', action='store_true')
    parser.add_argument('--profile-modes', nargs='+', default=['fp32', 'tf32'])
    args, remaining = parser.parse_known_args()
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from runner import batch_inference
    args.profile_root.mkdir(parents=True, exist_ok=True)
    allocations = Counter()

    class Allocations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, values=(), kwargs=None):
            if str(func).startswith(('aten.empty.', 'aten.empty_like.', 'aten.empty_strided.')):
                site = next((f for f in reversed(traceback.extract_stack(limit=30))
                             if '/flag_gems/' in f.filename), None)
                key = (str(func), site.filename if site else 'native',
                       site.lineno if site else 0, site.name if site else '')
                allocations[key] += 1
            return func(*values, **(kwargs or {}))

    original = batch_inference.get_default_runner
    def initialize(*values, **kwargs):
        runner = original(*values, **kwargs)
        predict = runner.predict
        calls = [0]
        def run(*values, **kwargs):
            index = calls[0]
            calls[0] += 1
            if index < len(args.profile_modes):
                return predict(*values, **kwargs)
            mode = args.profile_modes[index % len(args.profile_modes)]
            allocations.clear()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA]) as profile:
                with Allocations() if args.allocation_sites else nullcontext():
                    with torch.profiler.record_function('PREDICTION/' + mode):
                        result = predict(*values, **kwargs)
                    torch.cuda.synchronize()
            profile.export_chrome_trace(str(args.profile_root / (mode + '_trace.json')))
            rows = [{'name': e.key, 'count': e.count, 'self_cpu_us': e.self_cpu_time_total,
                     'self_device_us': e.self_device_time_total}
                    for e in profile.key_averages()]
            (args.profile_root / (mode + '_events.json')).write_text(json.dumps(rows, indent=2)+'\n')
            (args.profile_root / (mode + '_allocation_sites.json')).write_text(json.dumps([
                {'operator': key[0], 'source': key[1], 'line': key[2], 'function': key[3], 'count': count}
                for key, count in allocations.most_common()], indent=2)+'\n')
            return result
        runner.predict = run
        return runner
    batch_inference.get_default_runner = initialize
    sys.argv = [str(args.profile_entry), *remaining]
    runpy.run_path(str(args.profile_entry), run_name='__main__')


if __name__ == '__main__':
    main()
