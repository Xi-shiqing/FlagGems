"""Run the existing AF3Trainer; audit the actual precision and first update."""
import argparse
from collections import Counter
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing-tool', type=Path, required=True)
    parser.add_argument('--precision', choices=('fp32', 'tf32'), required=True)
    parser.add_argument('--candidate', default='public')
    parser.add_argument('--snapshot-reference', type=Path)
    args, remaining = parser.parse_known_args()
    for name in ('PROTENIX_EXTRA_SITE_PACKAGES',):
        if os.getenv(name):
            sys.path.append(os.environ[name])
    sys.modules.setdefault('wandb', types.ModuleType('wandb'))
    import torch
    from runner import train
    torch.set_num_threads(8)
    output = Path(remaining[remaining.index('--output-root') + 1])
    output.mkdir(parents=True, exist_ok=True)
    enabled = args.precision == 'tf32'
    ppu = remaining[remaining.index('--platform') + 1] == 'ppu810e'
    calls = Counter()
    state = {'trainer': None, 'initial': {}, 'gradients': {}, 'step_audited': False}
    native_env, native_step = train.AF3Trainer.init_env, train.AF3Trainer.train_step
    native_clip = torch.nn.utils.clip_grad_norm_
    import protenix.model.tri_attention as tri
    native_apply = tri.TriAttentionFunction.apply

    def apply(*values):
        # Official training kernel's deterministic flag selects IEEE vs TF32.
        values = (*values[:5], not enabled)
        calls['native_triattention_ieee' if not enabled else 'native_triattention_tf32'] += 1
        return native_apply(*values)

    if not ppu:
        tri.TriAttentionFunction.apply = staticmethod(apply)
    candidate = None
    if ppu:
        namespace = importlib.import_module('flag_gems.ops')
        module = importlib.import_module('flag_gems.ops.triangle_attention')
        public = module.triangle_attention
        if args.candidate != 'public':
            candidate = importlib.import_module(args.candidate)
        def operator(*values, **kwargs):
            calls['flaggems_triangle_' + str(kwargs.get('precision'))] += 1
            return (public if candidate is None else candidate.triangle_attention)(*values, **kwargs)
        namespace.triangle_attention = module.triangle_attention = operator

    def environment(self):
        native_env(self)
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled

    def clip(parameters, *a, **kw):
        trainer = state['trainer']
        if trainer is not None and not state['step_audited']:
            state['gradients'] = {name: p.grad.detach().float().cpu().clone()
                                  for name, p in trainer.model.named_parameters() if p.grad is not None}
        return native_clip(parameters, *a, **kw)

    def metrics(actual, reference):
        diff, gold = (actual.double() - reference.double()), reference.double()
        return {'difference_squared': float(diff.square().sum()), 'reference_squared': float(gold.square().sum()),
                'relative_l2': float(diff.norm() / gold.norm().clamp_min(1e-300)),
                'max_abs': float(diff.abs().max()), 'finite': bool(torch.isfinite(actual).all())}

    def step(self, batch):
        assert torch.backends.cuda.matmul.allow_tf32 == enabled
        first = not state['step_audited']
        if first:
            state['initial'] = {name: p.detach().float().cpu().clone() for name, p in self.model.named_parameters()}
        state['trainer'] = self
        result = native_step(self, batch)
        state['trainer'] = None
        if first:
            snapshot = {'gradients': state['gradients'],
                        'updates': {name: p.detach().float().cpu() - state['initial'][name] for name, p in self.model.named_parameters()},
                        'initial_sha256': {name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() for name, value in state['initial'].items()}}
            assert snapshot['gradients'], 'no actual model gradients captured'
            audit = {'scope': 'Full model parameter gradients before clipping, and actual first optimizer update; tensors stay on original server.',
                     'precision': args.precision, 'candidate': args.candidate, 'parameter_count': len(snapshot['updates']),
                     'gradient_parameter_count': len(snapshot['gradients']), 'reference': str(args.snapshot_reference), 'comparisons': {}}
            if args.snapshot_reference:
                reference = torch.load(args.snapshot_reference, map_location='cpu', weights_only=True)
                assert snapshot['initial_sha256'] == reference['initial_sha256'], 'initial model parameters differ'
                for kind in ('gradients', 'updates'):
                    assert snapshot[kind].keys() == reference[kind].keys()
                    rows = {name: metrics(value, reference[kind][name]) for name, value in snapshot[kind].items()}
                    numerator, denominator = [sum(row[key] for row in rows.values()) for key in ('difference_squared', 'reference_squared')]
                    audit['comparisons'][kind] = {'global_relative_l2': (numerator / max(denominator, 1e-300))**.5,
                        'global_max_abs': max(row['max_abs'] for row in rows.values()), 'finite': all(row['finite'] for row in rows.values()), 'parameters': rows}
            else:
                torch.save(snapshot, output / 'first_step_local_snapshot.pt')
            (output / 'actual_parameter_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
            state['initial'].clear()
            state['gradients'].clear()
            state['step_audited'] = True
        return result

    train.AF3Trainer.init_env = environment
    train.AF3Trainer.train_step = step
    torch.nn.utils.clip_grad_norm_ = clip
    sys.argv = [str(args.existing_tool), *remaining]
    try:
        runpy.run_path(str(args.existing_tool), run_name='__main__')
    finally:
        record = {'precision': args.precision, 'candidate': args.candidate, 'actual_calls': dict(calls),
                  'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                  'native_training_triangle_source': inspect.getfile(tri.TriAttentionFunction.forward),
                  'existing_tool_sha256': hashlib.sha256(args.existing_tool.read_bytes()).hexdigest(),
                  'candidate_source_sha256': hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest() if candidate else None,
                  'candidate_physical_backward_calls': getattr(candidate, '_PHYSICAL_BACKWARD_CALLS', 0)}
        (output / 'actual_training_entry.json').write_text(json.dumps(record, indent=2) + '\n')
        print('ACTUAL_TRAINING_ENTRY_AUDIT', record, flush=True)


if __name__ == '__main__':
    main()
