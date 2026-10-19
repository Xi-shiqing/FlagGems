"""Combine the two conditioning projections in AdaptiveLayerNorm at inference."""
import torch
import torch.nn.functional as F
from protenix.model.modules.primitives import AdaptiveLayerNorm

original_forward = AdaptiveLayerNorm.forward


def packed_forward(self, a, s):
    if self.training or a.device.type != 'musa' or torch.is_grad_enabled():
        return original_forward(self, a, s)
    a = self.layernorm_a(a)
    s = self.layernorm_s(s)
    wa, wb, bias = self.linear_s.weight, self.linear_nobias_s.weight, self.linear_s.bias
    dtype = torch.get_autocast_dtype('musa') if torch.is_autocast_enabled('musa') else s.dtype
    key = (id(wa), id(wb), id(bias), wa._version, wb._version, bias._version, dtype, s.device)
    if getattr(self, '_packed_key', None) != key:
        self._packed_ab = torch.cat((wa, wb), dim=0).to(dtype=dtype)
        self._packed_bias = torch.cat((bias, torch.zeros_like(bias)), dim=0).to(dtype=dtype)
        self._packed_key = key
    scale, shift = F.linear(s, self._packed_ab, self._packed_bias).chunk(2, dim=-1)
    return torch.sigmoid(scale) * a + shift


def install(model):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveLayerNorm) and 'diffusion_module' in name:
            module.forward = packed_forward.__get__(module, type(module))
            count += 1
    return count
