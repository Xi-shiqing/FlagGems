"""Inference-only joint A/B projection; native implementation remains fallback."""
import torch
import torch.nn.functional as F
from protenix.model.modules.primitives import Transition

original_forward = Transition.forward


def packed_forward(self, x):
    if self.training or x.device.type != 'musa' or torch.is_grad_enabled():
        return original_forward(self, x)
    # A and B see precisely the same normalized input. Concatenate weights
    # once for immutable evaluation weights; invalidate after any mutation.
    wa, wb = self.linear_no_bias_a.weight, self.linear_no_bias_b.weight
    dtype = torch.get_autocast_dtype('musa') if torch.is_autocast_enabled('musa') else x.dtype
    key = (id(wa), id(wb), wa._version, wb._version, dtype, x.device)
    if getattr(self, '_packed_key', None) != key:
        self._packed_ab = torch.cat((wa, wb), dim=0).to(dtype=dtype)
        self._packed_key = key
    shape = x.shape
    size = x.shape[-2]
    x = x.reshape(-1, shape[-1])
    chunks = torch.chunk(x, 1 if size < 3200 else 8, dim=-2)
    outputs = torch.empty((x.shape[0], self.c_in), dtype=x.dtype, device=x.device)
    start = 0
    for chunk in chunks:
        y = self.layernorm1(chunk)
        ab = F.linear(y, self._packed_ab)
        a, b = ab.chunk(2, dim=-1)
        F.silu(a, inplace=True)
        b.mul_(a)
        z = self.linear_no_bias(b)
        outputs[start:start+z.shape[0]] = z
        start += z.shape[0]
    return outputs.reshape(shape)


def install(model):
    # Start with trunk pair transitions; diffusion and confidence unchanged.
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Transition) and 'pairformer' in name and name.endswith('pair_transition'):
            module.forward = packed_forward.__get__(module, type(module))
            count += 1
    return count
