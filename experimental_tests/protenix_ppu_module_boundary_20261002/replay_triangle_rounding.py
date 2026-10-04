"""Replay actual triangle operands in memory; retain teacher predictions."""
import importlib
import inspect
import os
from functools import wraps
from audit_routes import metrics, rne, tensor_meta
from dense_triangle import references


def bit_statistics(value):
    import numpy as np
    array = value.detach().float().cpu().contiguous().numpy()
    bits = array.view(np.uint32)
    return {'numel': int(array.size), 'finite': bool(np.isfinite(array).all()),
            'zero_fraction': float(np.mean(array == 0)),
            'low_bits_zero_fraction': {str(count): float(np.mean((bits & ((1 << count) - 1)) == 0))
                                       for count in (7, 10, 13, 16)}}


def rtz(value):
    import numpy as np
    import torch
    array = value.detach().float().cpu().contiguous().numpy().copy()
    assert np.isfinite(array).all()
    bits = array.view(np.uint32)
    bits[:] &= np.uint32(0xffffe000)
    return torch.from_numpy(array)


def rna(value):
    import numpy as np
    import torch
    array = value.detach().float().cpu().contiguous().numpy().copy()
    assert np.isfinite(array).all()
    bits = array.view(np.uint32)
    bits[:] = (bits + np.uint32(0x1000)) & np.uint32(0xffffe000)
    return torch.from_numpy(array)


class TriangleReplay:
    def __init__(self, model):
        self.ppu = os.environ.get('PROTENIX_TRIATTENTION_FLAGGEMS_PUBLIC') == '1'
        self.namespace = importlib.import_module('flag_gems.ops' if self.ppu else 'cuequivariance_ops_torch')
        self.original = self.namespace.triangle_attention
        self.signature = inspect.signature(self.original)
        self.parameters = list(self.signature.parameters)[:3]
        self.mode = self.family = None
        self.rows = {}
        self.seen = set()
        self.forwards = []
        self.selected = []
        for name, module in model.named_modules():
            if type(module).__name__ != 'TriangleAttention':
                continue
            family = name.split('.')[0]
            self.selected.append({'module': name, 'family': family})
            original = module.forward
            def context(call, family, name):
                @wraps(call)
                def run(*args, **kwargs):
                    old = self.family
                    self.family = (family, name)
                    try:
                        return call(*args, **kwargs)
                    finally:
                        self.family = old
                return run
            module.forward = context(original, family, name)
            self.forwards.append((module, original))
        assert self.selected
        self.namespace.triangle_attention = self.replay

    def begin(self, mode):
        self.mode = mode
        self.rows[mode] = []
        self.seen.clear()

    def end(self):
        self.mode = None
        if 'tf32' in self.rows:
            self.namespace.triangle_attention = self.original
            for module, original in self.forwards:
                module.forward = original

    def replay(self, *args, **kwargs):
        import torch
        teacher = self.original(*args, **kwargs)
        if self.mode != 'tf32' or self.family is None or self.family[0] in self.seen:
            return teacher
        self.seen.add(self.family[0])
        bound = self.signature.bind(*args, **kwargs)
        operands = [bound.arguments[name] for name in self.parameters]
        assert all(value.dtype == torch.float32 for value in operands)
        previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        record = {'family': self.family[0], 'module': self.family[1],
                  'inputs': [tensor_meta(value) for value in operands],
                  'input_bit_statistics': [bit_statistics(value) for value in operands],
                  'teacher_output': tensor_meta(teacher), 'replays': {}}
        source = inspect.getsourcefile(self.original)
        record['entry_source'] = source
        modes = {'fp32': {}, 'tf32': {}}
        for rounding, convert in (('rne', rne), ('rtz', rtz), ('rna', rna)):
            rounded = {name: convert(value).to(value.device) for name, value in zip(self.parameters, operands)}
            for index, role in enumerate(('q', 'k', 'v')):
                modes[role + '_' + rounding] = {self.parameters[index]: rounded[self.parameters[index]]}
            modes['kv_' + rounding] = {name: rounded[name] for name in self.parameters[1:]}
            bias_name = 'bias2' if self.ppu else 'bias'
            modes['bias_' + rounding] = {bias_name: convert(bound.arguments[bias_name]).to(operands[0].device)}
        values = {}
        try:
            for mode, replacements in modes.items():
                precision = mode != 'fp32'
                torch.backends.cuda.matmul.allow_tf32 = precision
                torch.backends.cudnn.allow_tf32 = precision
                changed = self.signature.bind(*args, **kwargs)
                changed.arguments.update(replacements)
                if self.ppu:
                    changed.arguments['precision'] = 'tf32' if precision else 'ieee'
                value = self.original(*changed.args, **changed.kwargs)
                values[mode] = value.detach().cpu()
            assert tensor_meta(values['tf32']) == tensor_meta(teacher.detach().cpu()), 'teacher replay changed'
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            for name, value in references(bound, self.parameters, self.ppu):
                values[name] = value.detach().cpu()
            auxiliary = {}
            for mode in ('fp32', 'tf32'):
                torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
                torch.backends.cudnn.allow_tf32 = mode == 'tf32'
                changed = self.signature.bind(*args, **kwargs)
                if self.ppu:
                    impl = importlib.import_module('flag_gems.ops.triangle_attention')._streaming_triangle_attention_forward_impl
                    tensors = [changed.arguments[name] for name in list(self.signature.parameters)[:5]]
                    output, lse = impl(*tensors, precision='tf32' if mode == 'tf32' else 'ieee', save_lse=True)
                    auxiliary[mode] = {'output': output.detach().cpu(), 'lse': lse.detach().cpu()}
                else:
                    changed.arguments['return_aux'] = True
                    output, lse, maximum = self.original(*changed.args, **changed.kwargs)
                    auxiliary[mode] = {'output': output.detach().cpu(), 'lse': lse.detach().cpu(),
                                       'maximum': maximum.detach().cpu()}
            record['auxiliary'] = {'output_comparison_to_normal_entry': {
                mode: metrics(auxiliary[mode]['output'], values[mode]) for mode in auxiliary},
                'bit_statistics': {mode: {name: bit_statistics(value) for name, value in auxiliary[mode].items()
                                         if name != 'output'} for mode in auxiliary},
                'tf32_vs_fp32': {name: metrics(auxiliary['tf32'][name], auxiliary['fp32'][name])
                                  for name in auxiliary['tf32'] if name != 'output'}}
            if not self.ppu:
                from leaky_query_reference import references as query_references
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                record['leaky_query_decomposition'] = {}
                for name, output, lse, maximum in query_references(bound, self.parameters):
                    values[name] = output.detach().cpu()
                    record['leaky_query_decomposition'][name] = {
                        'lse_vs_native_tf32': metrics(lse.cpu(), auxiliary['tf32']['lse']),
                        'maximum_vs_native_tf32': metrics(maximum.cpu(), auxiliary['tf32']['maximum']),
                        'scope': 'Unmasked query bit addition per 16 keys; strict IEEE decomposition of the native diagnostic behavior, not a production candidate'}
            for mode, value in values.items():
                record['replays'][mode] = {'output_sha256': tensor_meta(value)['sha256'],
                    'vs_unmodified_tf32': metrics(value, values['tf32']),
                    'vs_unmodified_fp32': metrics(value, values['fp32']),
                    'output_hash_equal_unmodified_tf32': tensor_meta(value)['sha256'] == record['teacher_output']['sha256']}
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
        self.rows[self.mode].append(record)
        print('TRIANGLE_ROUNDING_REPLAY', self.family, flush=True)
        return teacher

    def close(self):
        assert self.rows.keys() == {'fp32', 'tf32'} and self.rows['tf32']
        return {'status': 'PASS', 'server': 'ppu' if self.ppu else 'h100', 'records': self.rows['tf32'],
                'scope': 'First actual triangle call per top-level semantic family. Quantization replays stay in server memory; only error/hash scalars persisted. Q probes also involve internal scaling. Hash idempotence is input-specific evidence, not proof for all inputs or ties.'}
