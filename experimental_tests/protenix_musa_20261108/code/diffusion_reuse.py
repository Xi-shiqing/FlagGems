"""Inference-only reuse of read-only diffusion conditioning, within one predict."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import inspect
import re
from textwrap import dedent, indent

import torch


def tensor_key(value):
    if torch.is_inference(value):
        raise RuntimeError("Reuse requires version-tracked tensors")
    return (value.data_ptr(), tuple(value.shape), value.stride(), value.dtype,
            value.device, value._version)


class DiffusionReuse:
    def __init__(self, model, trace):
        from protenix.model.modules.diffusion import DiffusionConditioning
        from protenix.model.modules.transformer import AttentionPairBias, AtomAttentionEncoder

        self.state = ContextVar("protenix_diffusion_reuse", default=None)
        self.counts = Counter()
        self.patches = {}
        self.diffusion_modules = {id(module) for module in model.diffusion_module.modules()}
        self.pair_modules = {id(module): module for module in model.diffusion_module.modules()
                             if isinstance(module, AttentionPairBias)}
        self.definition = torch.library.Library("flaggems_protenix_reuse", "DEF")
        self.definition.define("pair_bias(Tensor z, int module_id, int layout) -> Tensor")
        self.implementation = torch.library.Library("flaggems_protenix_reuse", "IMPL")

        def pair_bias(z, module_id, layout):
            module = self.pair_modules[module_id]
            assert z.dtype == module.linear_nobias_z.weight.dtype == torch.float32
            projected = module.linear_nobias_z(module.layernorm_z(z))
            head_first = (3, 0, 1, 2) if layout == 4 else (2, 0, 1)
            axes = list(range(projected.ndim - layout))
            return projected.permute(*axes, *(projected.ndim - layout + x for x in head_first))

        self.op = "flaggems_protenix_reuse::pair_bias"
        self.implementation.impl("pair_bias", lambda *args: trace(self.op, pair_bias, args, {}), "MUSA")

        def build_bias(module, z, layout, original):
            if not self.enabled(module):
                return original()
            return self.memo(module, "pair_bias", (z,), (layout,), lambda: (
                torch.ops.flaggems_protenix_reuse.pair_bias(z, id(module), layout)
            ))

        for method, axes in (("local_multihead_attention", "3, 0, 1, 2"),
                             ("standard_multihead_attention", "2, 0, 1")):
            original = getattr(AttentionPairBias, method)
            source = dedent(inspect.getsource(original))
            pattern = re.compile(
                rf"(?m)^( +)bias = self\.linear_nobias_z\(\s*self\.layernorm_z\(z\)"
                rf"\s*\)[^\n]*\n\1bias = permute_final_dims\(\s*bias, \[{axes}\]\s*\)[^\n]*"
            )
            layout = len(axes.split(","))
            source, count = pattern.subn(lambda match: match[1] +
                f"bias = _reuse_bias(self, z, {layout}, lambda: "
                f"permute_final_dims(self.linear_nobias_z(self.layernorm_z(z)), [{axes}]))", source)
            assert count == 1, f"Source changed: {method}"
            self.install(AttentionPairBias, original, source, _reuse_bias=build_bias)

        original = DiffusionConditioning.forward
        source = dedent(inspect.getsource(original))
        expected = "pair_z_clone = pair_z.clone()"
        assert source.count(expected) == 1
        source = source.replace(expected, "pair_z_clone = pair_z if _reuse_enabled(self) else pair_z.clone()")
        self.install(DiffusionConditioning, original, source, _reuse_enabled=self.enabled)

        original = AtomAttentionEncoder.forward
        source = dedent(inspect.getsource(original))
        start = source.index("    if p_lm is None or c_l is None:")
        end = source.index("    # Cross attention transformer", start)
        static = source[start:end]
        dynamic_start = static.index("        # Add the noisy positions")
        dynamic_end = static.index("    # Add the combined single conditioning", dynamic_start)
        static = static[:dynamic_start] + static[dynamic_end:]
        parameters = ", ".join(f"{name}={name}" for name in inspect.signature(original).parameters
                               if name != "self")
        replacement = (
            "    if not _reuse_enabled(self) or r_l is None or p_lm is None or c_l is None:\n"
            f"        return _original(self, {parameters})\n\n"
            "    def build_conditioning(p_lm=p_lm, c_l=c_l):\n"
            + indent(static.rstrip() + "\n    return n_token, c_l, p_lm\n", "    ")
            + "    n_token, c_l, p_lm = _reuse_atom(self, p_lm, c_l, s, atom_to_token_idx,\n"
              "                                       r_l is not None, inplace_safe, build_conditioning)\n"
              "    q_l = c_l + self.linear_no_bias_r(r_l) if r_l is not None else c_l.clone()\n\n"
        )
        self.install(AtomAttentionEncoder, original, source[:start] + replacement + source[end:],
                     _reuse_atom=self.atom_conditioning, _reuse_enabled=self.enabled,
                     _original=original)

    def install(self, cls, original, source, **bindings):
        name = f"{cls.__name__}.{original.__name__}"
        self.patches[name] = {
            "file": inspect.getsourcefile(original),
            "original_sha256": hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
            "patched_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "patched_source": source,
        }
        namespace = dict(original.__globals__, **bindings)
        exec(compile(source, f"<7bca-reuse:{name}>", "exec"), namespace)
        setattr(cls, original.__name__, namespace[original.__name__])

    def enabled(self, module):
        return (self.state.get() is not None and id(module) in self.diffusion_modules
                and not module.training and not torch.is_grad_enabled())

    def memo(self, module, tag, inputs, options, build):
        entries = self.state.get()
        key = tuple(tensor_key(value) for value in inputs), options
        slot = (id(module), tag)
        entry = entries.get(slot)
        if entry is not None and entry[0] == key:
            assert all(tensor_key(value) == saved for value, saved in entry[3])
            self.counts[tag + "_hit"] += 1
            return entry[2]
        result = build()
        outputs = result if isinstance(result, tuple) else (result,)
        tracked = [(value, tensor_key(value)) for value in outputs if isinstance(value, torch.Tensor)]
        entries[slot] = key, inputs, result, tracked  # Keep storages alive while keys exist.
        self.counts[tag + "_miss"] += 1
        return result

    def atom_conditioning(self, module, p_lm, c_l, s, atom_to_token, has_positions, inplace_safe, build):
        if not self.enabled(module) or not has_positions or p_lm is None or c_l is None:
            return build()
        return self.memo(module, "atom_conditioning", (p_lm, c_l, s, atom_to_token),
                         (inplace_safe,), build)

    @contextmanager
    def prediction(self):
        self.counts.clear()
        token = self.state.set({})
        try:
            yield
        finally:
            self.state.reset(token)
