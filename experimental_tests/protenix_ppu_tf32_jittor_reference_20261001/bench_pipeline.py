#!/usr/bin/env python3
"""Check effective compiler stages against native TF32 on the same real MM."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import time

import torch
import triton
import flag_gems
from bench_vendor_mm import Acblas, load_helper, round_buffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--capture', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--unsplit', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assert torch.utils.deterministic.fill_uninitialized_memory
    helper = load_helper(args.helper)
    module = importlib.import_module('flag_gems.ops.mm')
    original, stage, metadata = module.mm_direct_fused, [1], {}
    unsplit = [False]

    class StagedKernel:
        def __getitem__(self, grid):
            def launch(*a, **kw):
                kw['num_stages'] = stage[0] if kw['ALLOW_TF32'] else 1
                if unsplit[0] and kw['ALLOW_TF32']:
                    kw['SPLIT_K'] = 1
                result = original[grid](*a, **kw)
                key = ('unsplit' if unsplit[0] else 'stage') + str(stage[0])
                if key not in metadata:
                    details = getattr(result, 'metadata', None)
                    metadata[key] = {'split_k': kw['SPLIT_K'],
                        'shared_bytes': getattr(details, 'shared', None),
                        'num_registers': getattr(result, 'n_regs', None),
                        'num_spills': getattr(result, 'n_spills', None),
                        'ir_sha256': {k: hashlib.sha256(v.encode()).hexdigest()
                                      for k, v in getattr(result, 'asm', {}).items() if isinstance(v, str)}}
                return result
            return launch

    module.mm_direct_fused = StagedKernel()
    native = Acblas('/usr/local/PPU_SDK/targets/x86_64-linux/lib/libacblas.so')
    rows = []
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.inference_mode(), flag_gems.use_gems(exclude=[]):
            for capture_root in args.capture:
                record = json.loads((capture_root / 'activation_capture.json').read_text())['ops']['selected']['mm_large']
                a, b = helper._load_inputs(capture_root, record, 'cuda')
                assert a.is_contiguous() and b.T.is_contiguous()
                ar, br = torch.empty_like(a), torch.empty_like(b.T).T
                c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
                round_buffer[(triton.cdiv(a.numel(), 1024),)](a, ar, a.numel(), 1024)
                round_buffer[(triton.cdiv(b.numel(), 1024),)](b, br, b.numel(), 1024)
                modes = ('stage1', 'unsplit1', 'unsplit2', 'unsplit3', 'native_prepared') if args.unsplit else ('stage1', 'stage2', 'stage3', 'native_prepared')
                samples = {name: [] for name in modes}
                outputs, metadata = {}, {}
                for name in samples:
                    if name != 'native_prepared':
                        stage[0] = int(name[-1])
                        unsplit[0] = name.startswith('unsplit')
                        call = lambda: torch.mm(a, b)
                    else:
                        call = lambda: native.mm(ar, br, c, 77)
                    for _ in range(5):
                        y = call()
                    torch.cuda.synchronize()
                    outputs[name] = y.clone()
                if args.unsplit:
                    assert torch.equal(outputs['unsplit1'], outputs['unsplit2'])
                    assert torch.equal(outputs['unsplit1'], outputs['unsplit3'])
                    accumulation_error = helper._metrics(outputs['unsplit1'], outputs['stage1'])
                    assert accumulation_error['relative_l2'] < 1e-5 and accumulation_error['finite']
                else:
                    assert torch.equal(outputs['stage1'], outputs['stage2'])
                    assert torch.equal(outputs['stage1'], outputs['stage3'])
                    accumulation_error = None
                errors = helper._metrics(outputs['native_prepared'], outputs['stage1'])
                assert errors['relative_l2'] < 1e-5 and errors['finite']
                names = list(samples)
                for repeat in range(15):
                    order = names[repeat % len(names):] + names[:repeat % len(names)]
                    for name in order:
                        if name != 'native_prepared':
                            stage[0] = int(name[-1])
                            unsplit[0] = name.startswith('unsplit')
                            call = lambda: torch.mm(a, b)
                        else:
                            call = lambda: native.mm(ar, br, c, 77)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(3):
                            y = call()
                        torch.cuda.synchronize()
                        samples[name].append((time.perf_counter() - start) * 1000 / 3)
                row = {'capture': capture_root.name, 'inputs': [helper._metadata(t) for t in (a, b)],
                    'samples_wall_ms': samples, 'medians_wall_ms': {n: statistics.median(s) for n, s in samples.items()},
                    'compiled_metadata': metadata, 'stages_bitwise_equal_with_same_split': True,
                    'unsplit_vs_original': accumulation_error,
                    'native_rne_vs_gems_rne': errors}
                rows.append(row)
                print(json.dumps({k: row[k] for k in ('capture', 'medians_wall_ms', 'compiled_metadata')}), flush=True)
    finally:
        module.mm_direct_fused = original
        native.close()
    args.out.write_text(json.dumps({'status': 'PASS', 'rows': rows,
        'scope': 'FlagGems calls include output allocation; native_prepared excludes allocation and RNE preparation, and is a kernel reference only',
        'deterministic_algorithms': True, 'fill_uninitialized_memory': True,
        'unsplit_experiment': args.unsplit,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
