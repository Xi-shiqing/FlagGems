"""Group independent projections of the same activation before FP32 FlagGems Linear."""
import math

import torch
from torch.nn import functional as F


class PackedProjections:
    def __init__(self, model, trace):
        from protenix.model.modules.primitives import Attention
        from protenix.model.modules.transformer import ConditionedTransitionBlock

        self.definition = torch.library.Library("flaggems_protenix_packed", "DEF")
        self.definition.define("linear(Tensor x, Tensor weight, Tensor? bias) -> Tensor")
        self.implementation = torch.library.Library("flaggems_protenix_packed", "IMPL")
        self.op = "flaggems_protenix_packed::linear"
        self.implementation.impl("linear", lambda *args: trace(self.op, F.linear, args, {}), "MUSA")
        self.qkv = {}
        self.kv = {}
        self.transition = {}
        with torch.no_grad():
            for module in model.diffusion_module.modules():
                if isinstance(module, Attention):
                    self.kv[module] = self.pack(module.linear_k, module.linear_v)
                    if module.c_q == module.c_k == module.c_v:
                        self.qkv[module] = self.pack(module.linear_q, module.linear_k, module.linear_v)
                if isinstance(module, ConditionedTransitionBlock):
                    self.transition[module] = self.pack(module.linear_nobias_a1, module.linear_nobias_a2)
        original_attention = Attention._prep_qkv
        original_transition = ConditionedTransitionBlock.forward

        def attention(module, q_x, kv_x, apply_scale=True):
            if module not in self.kv or module.training or torch.is_grad_enabled():
                return original_attention(module, q_x, kv_x, apply_scale)
            if q_x is kv_x and module in self.qkv:
                q, k, v = self.project(q_x, self.qkv[module])
            else:
                q = module.linear_q(q_x)
                k, v = self.project(kv_x, self.kv[module])
            q, k, v = (part.view(part.shape[:-1] + (module.num_heads, -1)).transpose(-2, -3)
                       for part in (q, k, v))
            if apply_scale:
                q = q / math.sqrt(module.c_hidden)
            return q, k, v

        def transition(module, a, s):
            if module not in self.transition or module.training or torch.is_grad_enabled():
                return original_transition(module, a, s)
            a = module.adaln(a, s)
            gate, value = self.project(a, self.transition[module])
            hidden = torch.ops.flaggems_protenix.silu_and_mul(gate, value)
            return torch.sigmoid(module.linear_s(s)) * module.linear_nobias_b(hidden)

        Attention._prep_qkv = attention
        ConditionedTransitionBlock.forward = transition

    @staticmethod
    def pack(*layers):
        assert all(layer.weight.dtype == torch.float32 for layer in layers)
        weight = torch.cat([layer.weight for layer in layers])
        sizes = tuple(layer.out_features for layer in layers)
        bias = None
        if any(layer.bias is not None for layer in layers):
            bias = torch.cat([layer.bias if layer.bias is not None else weight.new_zeros(size)
                              for layer, size in zip(layers, sizes)])
        parameters = [parameter for layer in layers for parameter in (layer.weight, layer.bias)
                      if parameter is not None]
        return weight, bias, sizes, tuple((p, p._version) for p in parameters)

    @staticmethod
    def project(x, packed):
        weight, bias, sizes, parameters = packed
        assert x.dtype == torch.float32
        assert all(parameter._version == version for parameter, version in parameters)
        return torch.ops.flaggems_protenix_packed.linear(x, weight, bias).split(sizes, dim=-1)
