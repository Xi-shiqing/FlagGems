"""Export only timings/counts; do not sum nested CPU and device operators."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


def union_us(intervals):
    end, total = float('-inf'), 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize(path):
    events = json.loads(path.read_text())['traceEvents']
    kernels = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'kernel']
    copies = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('gpu_memcpy', 'gpu_memset')]
    launches = [e for e in events if e.get('ph') == 'X' and
                ('LaunchKernel' in e.get('name', '') or 'LaunchCooperativeKernel' in e.get('name', ''))]
    work = kernels + copies
    assert kernels, 'No kernel events: profiler device coverage unavailable'
    by_name = defaultdict(lambda: [0, 0.0])
    for event in kernels:
        by_name[event['name']][0] += 1
        by_name[event['name']][1] += event['dur']
    start = min(e['ts'] for e in work)
    stop = max(e['ts'] + e['dur'] for e in work)
    busy = union_us([(e['ts'], e['ts'] + e['dur']) for e in work])
    incomplete = len(kernels) != len(launches)
    return {
        'scope': 'Profiled diagnostic only; profiling changes host timing. Not acceptance wall time.',
        'kernel_events': len(kernels), 'launch_api_events': len(launches),
        'coverage_warning': 'Launch/kernel counts differ; device busy/idle are descriptive, not exact.' if incomplete else None,
        'profiled_device_span_ms': (stop-start)/1000,
        'observed_device_busy_union_ms': busy/1000,
        'uncovered_device_span_ms': (stop-start-busy)/1000,
        'observed_device_busy_pct': busy/(stop-start)*100,
        'gpu_copy_events': len(copies), 'gpu_copy_sum_ms': sum(e['dur'] for e in copies)/1000,
        'top_kernels': [{'name': name, 'count': values[0], 'sum_ms': values[1]/1000}
                        for name, values in sorted(by_name.items(), key=lambda row: -row[1][1])[:40]],
        'frequent_kernels': [{'name': name, 'count': values[0], 'sum_ms': values[1]/1000}
                             for name, values in sorted(by_name.items(), key=lambda row: -row[1][0])[:20]],
    }


if __name__ == '__main__':
    if sys.argv[1:] == ['--check']:
        assert union_us([(0, 4), (2, 6), (9, 10)]) == 7
        assert union_us([]) == 0
        print('TRACE_INTERVAL_CHECK_PASS')
    else:
        root = Path(sys.argv[1])
        result = {p.stem: summarize(p) for p in root.glob('*_trace.json')}
        assert result
        (root/'trace_summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: {field: value[field] for field in
              ('kernel_events','launch_api_events','coverage_warning','observed_device_busy_pct')}
              for k, value in result.items()}, indent=2))
