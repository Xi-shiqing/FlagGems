"""Capture one real Pairformer block per input, outside formal timing."""
import json
from pathlib import Path

import torch
from protenix.model.modules.pairformer import PairformerBlock
from protenix.model.modules.primitives import Transition


def install(folder):
    original = PairformerBlock.forward
    captured = set()
    handles = []
    def capture(module, args):
        x = args[0]
        key = (module.c_in, module.n, str(x.dtype))
        if key in activation_keys:
            return
        activation_keys.add(key)
        state = {k:v.detach().cpu() for k,v in module.state_dict().items()}
        sample = x.detach().reshape(-1, x.shape[-1])[:1024].contiguous().cpu()
        torch.save(dict(c_in=module.c_in, n=module.n, input_shape=list(x.shape),
                        dtype=str(x.dtype), activation_sample=sample, state_dict=state,
                        provenance='real trained model weights and first 1024 activation rows; full observed shape recorded'),
                   folder/f'transition_{module.c_in}_{module.n}_{str(x.dtype).split(".")[-1]}.pt')
    activation_keys = set()
    # Hook registration is passed the actual model by the caller.
    def forward(self, *args, **kwargs):
        key = (self.tri_mul_out.c_z, self.c_s)
        if key in captured:
            return original(self, *args, **kwargs)
        captured.add(key)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.PrivateUse1]
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            result = original(self, *args, **kwargs)
            torch.musa.synchronize()
        events = prof.key_averages(group_by_input_shape=True)
        prefix = f'pairformer_block_{key[0]}_{key[1]}_profile'
        (folder/(prefix+'.txt')).write_text(events.table(sort_by='self_device_time_total', row_limit=60))
        rows = []
        for e in events:
            rows.append(dict(name=e.key, count=e.count, input_shapes=e.input_shapes,
                             self_cpu_us=e.self_cpu_time_total, cpu_us=e.cpu_time_total,
                             self_device_us=e.self_device_time_total, device_us=e.device_time_total))
        (folder/(prefix+'.json')).write_text(json.dumps(rows, indent=2))
        return result
    PairformerBlock.forward = forward
    def uninstall():
        PairformerBlock.forward = original
        for handle in handles:
            handle.remove()
    def register(model):
        for module in model.modules():
            if isinstance(module, Transition):
                handles.append(module.register_forward_pre_hook(capture))
    uninstall.register = register
    return uninstall
