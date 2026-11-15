"""Optional event timings for nested stages of a separate MUSA diagnostic call."""
from collections import defaultdict
from functools import wraps

import torch


class StageProbe:
    def __init__(self, model):
        self.active = False
        self.events = defaultdict(list)
        self.token_shapes = defaultdict(int)
        self.token_step = 0
        self.token_active = False
        diffusion = model.diffusion_module
        self.wrap(model, "get_pairformer_output", "pairformer")
        self.wrap(model, "sample_diffusion", "diffusion")
        self.wrap(model.confidence_head, "forward", "confidence_head")
        for name in ("atom_attention_encoder", "diffusion_transformer", "atom_attention_decoder"):
            self.wrap(getattr(diffusion, name), "forward", name)

    def instrument_token(self, transformer):
        original = transformer.forward

        @wraps(original)
        def token(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            self.token_step += 1
            self.token_active = 2 <= self.token_step <= 5
            try:
                return original(*args, **kwargs)
            finally:
                self.token_active = False

        transformer.forward = token
        for block in transformer.blocks:
            self.wrap(block.attention_pair_bias, "forward", "token_sample.attention_total", token=True)
            self.wrap(block.conditioned_transition_block, "forward", "token_sample.transition_total", token=True)
            for name, module in block.named_modules():
                if isinstance(module, torch.nn.Linear):
                    self.wrap(module, "forward", f"token_sample.linear.{name}", token=True)

    def wrap(self, target, method, label, token=False):
        original = getattr(target, method)

        @wraps(original)
        def measured(*args, **kwargs):
            if not self.active or (token and not self.token_active):
                return original(*args, **kwargs)
            if label.startswith("token_sample.linear."):
                value = args[0]
                self.token_shapes[(label, tuple(value.shape), value.stride(), tuple(target.weight.shape))] += 1
            start = torch.musa.Event(enable_timing=True)
            end = torch.musa.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            self.events[label].append((start, end))
            return result

        setattr(target, method, measured)

    def read_synchronized(self):
        return {label: {"calls": len(pairs),
                        "elapsed_ms": [start.elapsed_time(end) for start, end in pairs]}
                for label, pairs in self.events.items()}

    def token_shape_receipt(self):
        return [
            {"name": name, "input_shape": shape, "input_stride": stride, "weight_shape": weight, "calls": calls}
            for (name, shape, stride, weight), calls in self.token_shapes.items()
        ]
