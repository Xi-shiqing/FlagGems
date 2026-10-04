"""Probe real TF32 activations without replacing the model's returned outputs."""
import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import runpy
import sys


def metrics(a, b):
    import torch
    with torch.no_grad():
        x, y = a.detach().float(), b.detach().float()
        delta = x - y
        return {'relative_l2': float(delta.norm() / y.norm().clamp_min(1e-30)),
                'max_abs': float(delta.abs().max()), 'signed_mean': float(delta.mean()),
                'rms': float(delta.square().mean().sqrt()),
                'finite': bool(torch.isfinite(x).all() and torch.isfinite(y).all())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shadow-harness', type=Path, required=True)
    p.add_argument('--shadow-output', type=Path, required=True)
    args, remaining = p.parse_known_args()
    extra = os.environ.get('PROTENIX_EXTRA_SITE_PACKAGES')
    if extra:
        sys.path.append(extra)
    import torch
    import flag_gems.ops as ops
    from torch.utils._python_dispatch import TorchDispatchMode
    from runner import batch_inference
    original_init = batch_inference.get_default_runner
    original_ta = ops.triangle_attention
    rows, ta_rows, seen = [], [], set()
    groups = Counter()
    attention_entries, attention_ops = Counter(), Counter()
    busy = [False]

    def triangle(*a, **kw):
        output = original_ta(*a, **kw)
        if busy[0] or len(ta_rows) >= 2:
            return output
        assert kw.get('precision') == 'tf32', kw
        busy[0] = True
        try:
            variants = {mode: original_ta(*a, **(kw | {'precision': mode}))
                        for mode in ('tf32', 'qk_tf32', 'pv_tf32', 'ieee')}
            record = {'call': len(ta_rows), 'input_shapes': [list(x.shape) for x in a],
                'precision': kw['precision'],
                'replay_vs_teacher': metrics(variants['tf32'], output),
                'vs_tf32': {mode: metrics(value, variants['tf32']) for mode, value in variants.items()},
                'vs_ieee': {mode: metrics(value, variants['ieee']) for mode, value in variants.items()}}
            assert record['replay_vs_teacher']['max_abs'] == 0
            ta_rows.append(record)
            print('TF32_SHADOW_TA', record, flush=True)
        finally:
            busy[0] = False
        return output

    def initialize(*a, **kw):
        runner = original_init(*a, **kw)
        for name, module in runner.model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            def probe(module, inputs, output, name=name):
                if busy[0] or name in seen or not isinstance(output, torch.Tensor):
                    return
                group = name.split('.', 1)[0]
                # Bound probe cost by module family, independently of input identity/shape.
                if groups[group] >= 12 or len(rows) >= 72 or output.dtype != torch.float32:
                    return
                seen.add(name)
                groups[group] += 1
                busy[0] = True
                previous = torch.backends.cuda.matmul.allow_tf32
                assert previous, 'teacher must use ordinary TF32'
                try:
                    with torch.no_grad():
                        tf = torch.nn.functional.linear(inputs[0], module.weight, module.bias)
                        torch.backends.cuda.matmul.allow_tf32 = False
                        strict = torch.nn.functional.linear(inputs[0], module.weight, module.bias)
                        record = {'module': name, 'input_shape': list(inputs[0].shape),
                            'weight_shape': list(module.weight.shape),
                            'tf32_replay_vs_teacher': metrics(tf, output),
                            'tf32_vs_ieee_on_same_input': metrics(tf, strict)}
                        assert record['tf32_replay_vs_teacher']['max_abs'] == 0
                        rows.append(record)
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = previous
                    busy[0] = False
            module.register_forward_hook(probe)
        return runner

    batch_inference.get_default_runner = initialize
    ops.triangle_attention = triangle
    sys.argv = [str(args.shadow_harness), *remaining]
    def profile(frame, event, arg):
        if event != 'call':
            return
        file, name = frame.f_code.co_filename, frame.f_code.co_name
        if ('flag_gems' in file and ('/attention.py' in file or '_scaled_dot_product' in file)) or (
                file.endswith('/modules/primitives.py') and name == '_attention'):
            meta = {'file': file, 'function': name,
                    'allow_tf32': torch.backends.cuda.matmul.allow_tf32}
            for key in ('q', 'query', 'use_efficient_implementation'):
                value = frame.f_locals.get(key)
                if isinstance(value, torch.Tensor):
                    meta[key] = {'shape': list(value.shape), 'dtype': str(value.dtype)}
                elif isinstance(value, bool):
                    meta[key] = value
            attention_entries[json.dumps(meta, sort_keys=True)] += 1

    class AttentionTrace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, a=(), kw=None):
            kw = kw or {}
            if 'scaled_dot_product' in str(func):
                record = {'op': str(func), 'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                    'inputs': [{'shape': list(x.shape), 'dtype': str(x.dtype)}
                               for x in a if isinstance(x, torch.Tensor)]}
                attention_ops[json.dumps(record, sort_keys=True)] += 1
            return func(*a, **kw)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with AttentionTrace():
            runpy.run_path(str(args.shadow_harness), run_name='__main__')
    finally:
        sys.setprofile(previous)
        ops.triangle_attention = original_ta
    assert rows and ta_rows
    args.shadow_output.write_text(json.dumps({'status': 'PASS', 'linear': rows, 'triangle': ta_rows,
        'attention_entries': [json.loads(k) | {'count': v} for k, v in attention_entries.items()],
        'attention_ops': [json.loads(k) | {'count': v} for k, v in attention_ops.items()],
        'policy': 'ordinary TF32 teacher; original outputs returned; identical operands for each local replay',
        'limits': 'PPU local decomposition only; this does not establish cross-device equality or causal RMSD attribution.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
