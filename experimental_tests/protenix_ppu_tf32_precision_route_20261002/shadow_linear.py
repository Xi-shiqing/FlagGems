"""Find early arithmetic divergence using real, hash-matched model inputs/weights."""
import argparse
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import time
from audit_routes import metrics, rne, tensor_meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--route-harness', type=Path, required=True)
    p.add_argument('--route-output', type=Path, required=True)
    args, remaining = p.parse_known_args()
    extra = os.environ.get('PROTENIX_EXTRA_SITE_PACKAGES')
    if extra:
        sys.path.append(extra)
    import torch
    from runner import batch_inference
    original_init = batch_inference.get_default_runner
    rows, seen = [], set()

    def initialize(*a, **kw):
        runner = original_init(*a, **kw)
        for name, module in runner.model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            def probe(module, inputs, teacher, name=name):
                if name in seen or len(rows) >= 4:
                    return
                seen.add(name)
                x, w, b = inputs[0], module.weight, module.bias
                previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
                assert previous[0], 'ordinary TF32 teacher required'
                replay, modes = {}, {}
                try:
                    with torch.no_grad():
                        for mode in ('fp32', 'tf32'):
                            torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
                            torch.backends.cudnn.allow_tf32 = mode == 'tf32'
                            for _ in range(3):
                                torch.nn.functional.linear(x, w, b)
                            torch.cuda.synchronize()
                            samples = []
                            for _ in range(10):
                                start = time.perf_counter()
                                value = torch.nn.functional.linear(x, w, b)
                                torch.cuda.synchronize()
                                samples.append((time.perf_counter()-start)*1000)
                            replay[mode] = value
                            modes[mode] = {'median_ms': statistics.median(samples), 'samples_ms': samples,
                                           'output_sha256': tensor_meta(value)['sha256']}
                        xx = x.detach().cpu().reshape(-1, x.shape[-1])[:8]
                        ww = w.detach().cpu()[:8]
                        gold = xx.double() @ ww.double().T
                        rounded = rne(xx).double() @ rne(ww).double().T
                        if b is not None:
                            gold += b.detach().cpu()[:8].double()
                            rounded += b.detach().cpu()[:8].double()
                        for mode in ('fp32', 'tf32'):
                            sample = replay[mode].detach().cpu().reshape(-1, w.shape[0])[:8, :8]
                            modes[mode]['sample_vs_ieee_fp64'] = metrics(sample, gold)
                            modes[mode]['sample_vs_rne_tf32_fp64'] = metrics(sample, rounded)
                            torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
                            torch.backends.cudnn.allow_tf32 = mode == 'tf32'
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                                torch.nn.functional.linear(x, w, b)
                                torch.cuda.synchronize()
                            modes[mode]['kernel_trace'] = [e.name for e in prof.events() if 'CUDA' in str(e.device_type)]
                        record = {'module': name, 'inputs': [tensor_meta(x), tensor_meta(w)],
                                  'bias': tensor_meta(b) if b is not None else None,
                                  'tf32_vs_fp32': metrics(replay['tf32'], replay['fp32']),
                                  'tf32_replay_vs_teacher': metrics(replay['tf32'], teacher), 'modes': modes}
                        assert record['tf32_replay_vs_teacher']['max_abs'] == 0
                        rows.append(record)
                        print('EARLY_LINEAR', name, list(x.shape), record['tf32_vs_fp32'], flush=True)
                finally:
                    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
            module.register_forward_hook(probe)
        return runner

    batch_inference.get_default_runner = initialize
    sys.argv = [str(args.route_harness), *remaining]
    runpy.run_path(str(args.route_harness), run_name='__main__')
    assert len(rows) == 4
    args.route_output.write_text(json.dumps({'status': 'PASS', 'linear': rows,
        'scope': 'First four executed Linear modules; original TF32 teacher outputs returned. Timings are local identical-operand replays, not E2E measurements.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
