"""Traceable inference-only integration of the frozen Xi-shiqing FlagGems tree.

No native retry is performed. Unavailable implementations are reported before
registration. The original repository stays unchanged; compatibility overlays
are copied sources with explicit hashes and diffs.
"""
import ast
from collections import Counter
import hashlib
import importlib
import importlib.machinery
import importlib.util
import inspect
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import types
import traceback
import time

import torch
import flag_gems

class RepositoryAdapter:
    def __init__(self, root, *, sdpa=True, linear_mm=True, silu_fusion=True, excluded=(), vendor_group="all", artifact_root=None):
        self.root = Path(root)
        self.artifact_root = Path(artifact_root) if artifact_root is not None else self.root
        self.vendor_group = vendor_group
        self.repository = self.root / 'FlagGems'
        assert str(self.repository / 'src') in flag_gems.__file__, flag_gems.__file__
        assert all((self.root / 'results/flaggems_baseline_compat3' / task / 'complete.json').exists()
                   for task in ['7bca', '6zsv', '7bb8']), 'Baseline must finish first'
        commit = subprocess.check_output(['git', '-C', str(self.repository), 'rev-parse', 'HEAD'], text=True).strip()
        assert commit == 'bfeca796e162b61302abb44b1b95af151335f08d'
        self.excluded = set(excluded)
        self.trace = None
        self.custom_keys = []
        self.vendor = {}
        self.info = {'repository': 'https://github.com/Xi-shiqing/FlagGems', 'commit': commit,
                     'actual_package_file': flag_gems.__file__,
                     'reported_package_version': flag_gems.__version__,
                     'version_note': 'module may report old installed distribution metadata; source commit/path/hashes identify actual code',
                     'scope': 'FP32 inference; no training acceptance; no native retry',
                     'overlays': [], 'module_load_failures': {}, 'registration_choices': {},
                     'vendor_group': vendor_group, 'explicit_exclusions': sorted(self.excluded), 'model_fusions': []}
        observed = set()
        for task in ['7bca', '6zsv', '7bb8']:
            p = self.root / 'results/flaggems_baseline_compat3' / task / 'execution_audit.json'
            observed.update(json.loads(p.read_text())['completed_function_calls'])
        self.observed = observed
        requested_names = {item[1].__name__ for item in flag_gems._FULL_CONFIG if item[0] in observed}
        if linear_mm:
            requested_names.add('mm')
        folder = self.repository / 'src/flag_gems/runtime/backend/_mthreads/ops'
        self.overlay = self.artifact_root / 'flaggems_repo_overlays'
        self.overlay.mkdir(exist_ok=True)
        # Optional FP8 TensorDescriptor imports in the upstream package init
        # prevent every backend op from loading on this vendor Triton build.
        # Give normal Python child imports a package path without running the
        # all-or-nothing init; import each relevant implementation explicitly.
        name = 'flag_gems.runtime.backend._mthreads.ops'
        package = types.ModuleType(name)
        package.__path__ = [str(folder)]
        package.__package__ = name
        package.__file__ = str(folder / '__init__.py')
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
        parent = importlib.import_module('flag_gems.runtime.backend._mthreads')
        parent.ops = package
        modules = {}
        definitions = {}
        for path in sorted(folder.glob('*.py')):
            if path.name == '__init__.py':
                continue
            names = {n.name for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
            relevant = names & requested_names
            if not relevant:
                continue
            try:
                module = self._load_backend(path, name)
                modules[path.stem] = module
                for function_name in relevant:
                    self.vendor[function_name] = getattr(module, function_name)
                    definitions[function_name] = str(path)
            except Exception as error:
                self.info['module_load_failures'][str(path)] = traceback.format_exc()
        self.info['available_backend_definitions'] = definitions
        self.sdpa_enabled = sdpa and 'sdpa' not in self.excluded
        self.linear_mm_enabled = linear_mm and 'linear_mm' not in self.excluded
        self.silu_enabled = silu_fusion and 'silu_fusion' not in self.excluded
        bootstrap = self.artifact_root / 'repository_bootstrap_reports'
        bootstrap.mkdir(exist_ok=True)
        self.write(bootstrap / ('inspection_' + str(time.time_ns()) + '.json'))
        if self.linear_mm_enabled:
            assert 'mm' in self.vendor, 'Required repository mm implementation unavailable'
        if self.sdpa_enabled:
            self.sdpa_module = self._load_strict_sdpa()
            self.sdpa_function = self.sdpa_module._scaled_dot_product_attention_math
            flag_gems._FULL_CONFIG = tuple(flag_gems._FULL_CONFIG) + (
                ('_scaled_dot_product_attention_math_musa', self.math_sdpa),)

    def _load_backend(self, original, package):
        name = package + '.' + original.stem
        source = original.read_text()
        descriptor_import = 'from triton.tools.tensor_descriptor import TensorDescriptor'
        path = original
        if descriptor_import in source:
            try:
                importlib.import_module('triton.tools.tensor_descriptor')
            except ModuleNotFoundError:
                patched = source.replace(descriptor_import, 'TensorDescriptor = None  # unavailable; FP32 FMA path only')
                patched = re.sub(r'^SQMMA_ON = .*$', 'SQMMA_ON = False  # TensorDescriptor unavailable', patched, flags=re.M)
                # libtuner hashes every declared JIT body at import time, even
                # when FP32 will never select SQMMA. Keep the FMA declarations
                # intact and make only unavailable descriptor paths explicit.
                lines = patched.splitlines(keepends=True)
                omitted = []
                for node in ast.parse(patched).body:
                    if isinstance(node, ast.FunctionDef) and 'sqmma' in node.name and node.name != 'is_sqmma_compatible':
                        start = min([node.lineno] + [x.lineno for x in node.decorator_list]) - 1
                        end = node.end_lineno
                        lines[start] = 'def ' + node.name + '(*args, **kwargs):\n'
                        lines[start + 1] = "    raise NotImplementedError('Descriptor/SQMMA API unavailable; FP32 FMA integration only')\n"
                        for index in range(start + 2, end):
                            lines[index] = '\n'
                        omitted.append(node.name)
                patched = ''.join(lines)
                path = self.overlay / '_mthreads/ops' / original.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(patched)
                for yaml in original.parent.parent.glob('*.yaml'):
                    link = path.parent.parent / yaml.name
                    if not link.exists():
                        link.write_bytes(yaml.read_bytes())
                self.info['overlays'].append({'original': str(original), 'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
                    'overlay': str(path), 'overlay_sha256': hashlib.sha256(patched.encode()).hexdigest(),
                    'change': 'unavailable descriptor/SQMMA declarations disabled before JIT hashing; FP32 existing FMA kernel unchanged',
                    'disabled_unavailable_functions': omitted})
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module

    def _load_strict_sdpa(self):
        original = self.repository / 'src/flag_gems/ops/_scaled_dot_product_attention_math.py'
        source = original.read_text()
        replacements = {
            'tl.dot(query, tl.trans(key))': 'tl.dot(query, tl.trans(key), allow_tf32=False)',
            'tl.dot(p.to(value.dtype), value)': 'tl.dot(p.to(value.dtype), value, allow_tf32=False)'}
        patched = source
        for old, new in replacements.items():
            assert old in patched
            patched = patched.replace(old, new)
        path = self.overlay / 'strict_fp32_sdpa_math.py'
        path.write_text(patched)
        self.info['overlays'].append({'original': str(original), 'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'overlay': str(path), 'overlay_sha256': hashlib.sha256(patched.encode()).hexdigest(),
            'change': 'existing repository SDPA kernel; explicitly disable TF32 in both dot products'})
        name = 'flag_gems._protenix_strict_fp32_sdpa_math'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def override(self, key, function):
        if key in self.excluded or function.__name__ in self.excluded:
            return function
        selected = function
        if key == 'linear' and self.linear_mm_enabled:
            selected = self.linear
        elif (function.__name__ in self.vendor
              and (key in self.observed or key == 'mm')
              and (self.vendor_group == 'all'
                   or self.vendor_group == 'bmm' and key.split('.')[0] == 'bmm'
                   or self.vendor_group == 'other' and key.split('.')[0] != 'bmm')):
            selected = self.vendor[function.__name__]
        if selected is not function or key == '_scaled_dot_product_attention_math_musa':
            self.info['registration_choices'][key] = {
                'function': getattr(selected, '__qualname__', repr(selected)),
                'code_file': getattr(getattr(selected, '__code__', None), 'co_filename', None),
                'baseline_function': function.__name__}
        return selected

    def linear(self, value, weight, bias=None):
        assert value.dtype == weight.dtype == torch.float32, 'FP32-only linear integration'
        assert value.shape[-1] == weight.shape[-1]
        result = torch.mm(value.reshape(-1, value.shape[-1]), weight.transpose(0, 1))
        if bias is not None:
            result = result + bias
        return result.reshape(*value.shape[:-1], weight.shape[0])

    def math_sdpa(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None):
        assert query.dtype == key.dtype == value.dtype == torch.float32
        assert dropout_p == 0.0, 'Only inference dropout=0 supported'
        assert query.shape[-1] == key.shape[-1] == value.shape[-1]
        leading = torch.broadcast_shapes(query.shape[:-2], key.shape[:-2], value.shape[:-2])
        qlen, klen, dim = query.shape[-2], key.shape[-2], query.shape[-1]
        heads = leading[-1] if leading else 1
        batch = math.prod(leading[:-1]) if leading else 1
        q = query.expand(*leading, qlen, dim).reshape(batch, heads, qlen, dim)
        k = key.expand(*leading, klen, dim).reshape(batch, heads, klen, dim)
        v = value.expand(*leading, klen, dim).reshape(batch, heads, klen, dim)
        mask = attn_mask
        if mask is not None:
            if mask.dtype == torch.bool:
                mask = torch.where(mask, 0.0, float('-inf')).to(query.dtype)
            mask = mask.expand(*leading, qlen, klen).reshape(batch, heads, qlen, klen)
        if qlen == 0 or klen == 0:
            out = torch.zeros((*leading, qlen, dim), dtype=query.dtype, device=query.device)
            weights = torch.empty((*leading, qlen, klen), dtype=query.dtype, device=query.device)
        else:
            out, weights = self.sdpa_function(q, k, v, mask, 0.0, is_causal, scale=scale)
            out = out.reshape(*leading, qlen, dim)
            weights = weights.reshape(*leading, qlen, klen)
        # The MUSA schema has a third dropout-mask output. No mask is used for
        # dropout=0 in the no-grad inference path; training is out of scope.
        return out, weights, torch.empty((0,), dtype=torch.bool, device=query.device)

    def install_model_fusions(self, model):
        if not self.silu_enabled:
            return
        self.fused_definition = torch.library.Library('flaggems_protenix', 'DEF')
        self.fused_definition.define('silu_and_mul(Tensor a, Tensor b) -> Tensor')
        self.fused_implementation = torch.library.Library('flaggems_protenix', 'IMPL')
        implementation = flag_gems.silu_and_mul
        def fused(a, b):
            assert a.dtype == b.dtype == torch.float32
            if self.trace is not None:
                return self.trace('flaggems_protenix::silu_and_mul', implementation, (a, b), {})
            return implementation(a, b)
        self.fused_implementation.impl('silu_and_mul', fused, 'MUSA')
        self.custom_keys.append('flaggems_protenix::silu_and_mul')
        from protenix.model.modules.primitives import Transition
        from protenix.model.modules.transformer import ConditionedTransitionBlock
        originals = {Transition: Transition.forward, ConditionedTransitionBlock: ConditionedTransitionBlock.forward}
        def transition(module, x):
            assert not module.training, 'Inference-only model integration'
            other_dims, dim_size, size = x.shape[:-1], x.shape[-1], x.shape[-2]
            x = x.reshape(-1, dim_size)
            chunks = torch.chunk(x, 1 if size < 3200 else 8, dim=-2)
            outputs = torch.empty((x.shape[0], module.c_in), dtype=x.dtype, device=x.device)
            start = 0
            for chunk in chunks:
                y = module.layernorm1(chunk)
                a = module.linear_no_bias_a(y)
                b = module.linear_no_bias_b(y)
                del y
                b = torch.ops.flaggems_protenix.silu_and_mul(a, b)
                del a
                b = module.linear_no_bias(b)
                outputs[start:start + b.shape[0]] = b
                start += b.shape[0]
                del b
            return outputs.reshape(*other_dims, module.c_in)
        def conditioned(module, a, s):
            assert not module.training, 'Inference-only model integration'
            a = module.adaln(a, s)
            b = torch.ops.flaggems_protenix.silu_and_mul(module.linear_nobias_a1(a), module.linear_nobias_a2(a))
            return torch.sigmoid(module.linear_s(s)) * module.linear_nobias_b(b)
        for cls, replacement in [(Transition, transition), (ConditionedTransitionBlock, conditioned)]:
            instances = sum(isinstance(m, cls) for m in model.modules())
            self.info['model_fusions'].append({'class': cls.__module__ + '.' + cls.__name__, 'instances': instances,
                'original_forward': inspect.getsource(originals[cls]), 'replacement_forward': inspect.getsource(replacement),
                'repository_fused_function_file': str(self.repository / 'src/flag_gems/fused/silu_and_mul.py')})
            cls.forward = replacement

    def write(self, path):
        Path(path).write_text(json.dumps(self.info, indent=2))
