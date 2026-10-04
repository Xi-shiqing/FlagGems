"""Regenerate identical synthetic operands with recorded real-model shapes/strides."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from audit_routes import metrics, output, rne, tensor_meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', choices=('ppu', 'h100'), required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    extra = os.environ.get('PROTENIX_EXTRA_SITE_PACKAGES')
    if extra:
        sys.path.append(extra)
    import numpy as np
    import torch
    if args.server == 'ppu':
        import flag_gems
    torch.set_num_threads(8)

    def make(meta, seed):
        shape, stride = meta['shape'], meta['stride']
        extent = 1 + sum((n-1)*s for n, s in zip(shape, stride))
        x = np.arange(extent, dtype=np.uint32) + np.uint32(seed)
        x ^= x >> 16
        x *= np.uint32(0x7feb352d)
        x ^= x >> 15
        x *= np.uint32(0x846ca68b)
        x ^= x >> 16
        data = ((x >> 8).astype(np.float32) * np.float32(2**-23) - np.float32(1)) * np.float32(.25)
        cpu = torch.as_strided(torch.from_numpy(data), shape, stride)
        gpu = torch.as_strided(torch.from_numpy(data).cuda(), shape, stride)
        assert tuple(gpu.stride()) == tuple(stride)
        return cpu, gpu

    records = []
    specs = json.loads(args.spec.read_text())
    for case_idx, spec in enumerate(specs):
        pairs = [make(meta, 101+case_idx*17+i) for i, meta in enumerate(spec['inputs'])]
        cpu, gpu = [x[0] for x in pairs], [x[1] for x in pairs]
        if spec['op'] == 'bmm':
            fn = lambda: torch.bmm(*gpu)
            gold = cpu[0].double() @ cpu[1].double()
            rounded_gold = rne(cpu[0]).double() @ rne(cpu[1]).double()
        else:
            fn = lambda: output(torch.ops.aten._scaled_dot_product_efficient_attention.default(
                *gpu, False, 0., False, scale=1.))
            q, k, v, bias = cpu
            probability = torch.softmax(q.double() @ k.double().transpose(-1, -2) + bias.double(), dim=-1)
            gold = probability @ v.double()
            rp = torch.softmax(rne(q).double() @ rne(k).double().transpose(-1, -2) + bias.double(), dim=-1)
            rounded_gold = rne(rp.float()).double() @ rne(v).double()
        record = {'name': spec['name'], 'op': spec['op'],
                  'inputs': [tensor_meta(x) for x in gpu],
                  'gold_ieee_fp64_sha256': tensor_meta(gold)['sha256'],
                  'gold_rne_tf32_fp64_sha256': tensor_meta(rounded_gold)['sha256'],
                  'modes': {}}
        values = {}
        with flag_gems.use_gems(exclude=[]) if args.server == 'ppu' else nullcontext():
            for mode in ('fp32', 'tf32'):
                torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
                torch.backends.cudnn.allow_tf32 = mode == 'tf32'
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                samples = []
                for _ in range(10):
                    start = time.perf_counter()
                    value = fn()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter()-start)*1000)
                values[mode] = value.detach().cpu()
                record['modes'][mode] = {'median_ms': statistics.median(samples), 'samples_ms': samples,
                    'output_sha256': tensor_meta(value)['sha256'],
                    'vs_ieee_fp64': metrics(values[mode], gold),
                    'vs_rne_tf32_fp64': metrics(values[mode], rounded_gold)}
            record['tf32_vs_fp32'] = metrics(values['tf32'], values['fp32'])
            for mode in ('fp32', 'tf32'):
                torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
                torch.backends.cudnn.allow_tf32 = mode == 'tf32'
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    fn()
                    torch.cuda.synchronize()
                record['modes'][mode]['kernel_trace'] = [e.name for e in prof.events() if 'CUDA' in str(e.device_type)]
        records.append(record)
        print('SAME_OPERAND_PROBE', record['name'], record['tf32_vs_fp32'], flush=True)
        args.output.write_text(json.dumps({'server': args.server, 'records': records,
            'scope': 'Deterministic synthetic operands; real-model shapes and strides. Native Efficient SDPA remains native on PPU.',
            'protocol': 'seed 101 integer mixer; float32 storage; per mode 3 warmups and 10 synchronized wall timings; separate kernel trace; full CPU FP64 gold.',
            'status': 'PASS' if len(records) == len(specs) else 'PARTIAL'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
