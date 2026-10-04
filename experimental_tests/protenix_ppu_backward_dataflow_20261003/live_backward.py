"""Operator gradient/update and timing probes on real in-memory model activations."""
import hashlib
import importlib
import json
import statistics
import time


def audit_live_training(owner, bound, q, k, v, bias1, bias2, scale):
    import numpy as np
    import torch
    from bench_query_once import metrics, sha, LR
    from audit_routes import tensor_meta
    prefix = '' if owner.ppu else 'cuda_'
    parent = importlib.import_module(prefix + 'fixed_rne_once')
    modules = [('fp32', None, False), ('tf32', None, True)]
    if not owner.ppu:
        modules.append(('custom_parent_rne', parent, True))
    modules.extend((name, module, True) for name, module in owner.candidates.items())
    family = owner.family[0]
    seed = 101 + int(hashlib.sha256(family.encode()).hexdigest()[:6], 16)
    rng = np.random.Generator(np.random.PCG64(seed))
    upstream_array = rng.integers(-32768, 32768, size=tuple(q.shape), dtype=np.int32).astype(np.float32) * 2.0**-16
    groups = sorted(set((0, q.shape[1]-1)))
    with torch.inference_mode(False), torch.enable_grad():
        params = [value.detach().clone().requires_grad_() for value in (q, k, v, bias2)]
        initial = [value.detach().clone() for value in params]
        bias1 = bias1.detach().clone()
        native_mask = bound.arguments.get('mask')
        if native_mask is not None:
            native_mask = native_mask.detach().clone()
        upstream = torch.from_numpy(upstream_array).to(q.device)
        sample_params = [value[:, groups].detach().cpu().double().requires_grad_() for value in params[:3]]
        a, b, c = [value.permute(0, 1, 3, 2, 4) for value in sample_params]
        bias = bias2.detach().cpu().double()
        mask_bias = bias1[:, groups].detach().cpu().double()
        gold = (((a @ b.transpose(-1, -2)) * scale + bias + mask_bias).softmax(-1) @ c).permute(0, 1, 3, 2, 4)
        gold_grads = torch.autograd.grad(gold, sample_params, upstream[:, groups].cpu().double())
        record = {'seed': seed, 'rng': 'NumPy PCG64 on CPU; original model activations are unchanged',
                  'upstream_sha256': sha(upstream_array), 'raw_parameter_inputs': [tensor_meta(value) for value in params],
                  'scope': 'Full live-operator Q/K/V/bias gradients and SGD input updates, not model-weight training. CPU FP64 Q/K/V gold samples two N groups, retaining every query/key and D. Bias gradient compared fully to the RNE parent. Native CUE uses raw Q and scale after QK; custom H100 scales Q before rounding and remains separately labeled.',
                  'gold_groups': groups, 'warmups': 3, 'repeats': 10, 'learning_rate': LR, 'modes': {}}

        def run(module, enabled):
            torch.backends.cuda.matmul.allow_tf32 = enabled
            torch.backends.cudnn.allow_tf32 = enabled
            if module is not None:
                scaled_q = params[0] if scale == 1. else params[0] * scale
                return module.triangle_attention(scaled_q, *params[1:3], bias1, params[3], precision='tf32' if enabled else 'ieee', layout='BNSHD')
            if owner.ppu:
                return owner.original(*params[:3], bias1, params[3], precision='tf32' if enabled else 'ieee', layout='BNSHD')
            changed = owner.signature.bind(*bound.args, **bound.kwargs)
            for name, param in zip(owner.parameters, params[:3]):
                value = param.permute(0, 1, 3, 2, 4)
                if bound.arguments[name].ndim == 4:
                    value = value.squeeze(0)
                changed.arguments[name] = value
            changed.arguments['bias'] = params[3].squeeze(0) if bound.arguments['bias'].ndim == 4 else params[3]
            if native_mask is not None:
                changed.arguments['mask'] = native_mask
            changed.arguments['return_aux'] = True
            value = owner.original(*changed.args, **changed.kwargs)[0]
            if value.ndim == 4:
                value = value.unsqueeze(0)
            return value.permute(0, 1, 3, 2, 4)

        def reset():
            with torch.no_grad():
                for param, value in zip(params, initial):
                    param.copy_(value)

        references = {}
        # Parent RNE is the comparison target, never a replacement for native baselines.
        output = run(parent, True)
        parent_grads = torch.autograd.grad(output, params, upstream)
        references = {key: value.detach().cpu() for key, value in zip(('q', 'k', 'v', 'bias2'), parent_grads)}
        parent_output = output.detach().cpu()
        del output, parent_grads
        for name, module, enabled in modules:
            reset()
            output = run(module, enabled)
            row = {'classification': 'original native/public' if module is None else 'custom kernel',
                   'output_sha256': sha(output.detach().cpu().numpy()), 'output_sample_vs_fp64': metrics(output[:, groups], gold),
                   'output_vs_parent_rne': metrics(output, parent_output), 'parameters': {}, 'timing_samples': {'inference': [], 'training': []}}
            try:
                grads = torch.autograd.grad(output, params, upstream)
                row['training_status'] = 'SUPPORTED'
            except RuntimeError as error:
                if owner.ppu or name != 'fp32' or 'Full precision FP32 backward pass for triangle attention is not implemented' not in str(error):
                    raise
                grads = ()
                row['training_status'] = 'NATIVE_FP32_BACKWARD_UNSUPPORTED'
            for index, (key, param, grad) in enumerate(zip(('q', 'k', 'v', 'bias2'), params, grads)):
                updated = param.detach().clone().add_(grad, alpha=-LR)
                parent_updated = param.detach().cpu().clone().add_(references[key], alpha=-LR)
                row['parameters'][key] = {'gradient_vs_parent_rne': metrics(grad, references[key]),
                    'realized_update_vs_parent_rne': metrics(updated-param.detach(), parent_updated-param.detach().cpu()),
                    'gradient_sha256': sha(grad.detach().cpu().numpy())}
                if index < 3:
                    row['parameters'][key]['gradient_sample_vs_fp64'] = metrics(grad[:, groups], gold_grads[index])
                if name in owner.candidates:
                    assert row['parameters'][key]['gradient_vs_parent_rne']['relative_l2'] < 1e-4
                    assert row['parameters'][key]['gradient_vs_parent_rne']['finite']
            if name in owner.candidates:
                assert row['output_vs_parent_rne']['max_abs'] == 0.
            record['modes'][name] = row
            del output, grads
        for phase in ('inference', 'training'):
            def execute(module, enabled):
                if phase == 'inference':
                    with torch.no_grad():
                        run(module, enabled)
                    return
                output = run(module, enabled)
                grads = torch.autograd.grad(output, params, upstream)
                with torch.no_grad():
                    for param, grad in zip(params, grads):
                        param.add_(grad, alpha=-LR)
            available = [mode for mode in modules if phase == 'inference' or record['modes'][mode[0]]['training_status'] == 'SUPPORTED']
            for _, module, enabled in available:
                for _ in range(3):
                    reset()
                    execute(module, enabled)
            for repeat in range(10):
                for name, module, enabled in available if repeat % 2 == 0 else reversed(available):
                    reset()
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin = time.perf_counter()
                    start.record()
                    execute(module, enabled)
                    end.record()
                    end.synchronize()
                    record['modes'][name]['timing_samples'][phase].append({'event_ms': start.elapsed_time(end), 'wall_ms': 1000*(time.perf_counter()-begin)})
            for row in record['modes'].values():
                samples = row['timing_samples'][phase]
                row.setdefault('timing_median', {})[phase] = {key: statistics.median(value[key] for value in samples) for key in ('event_ms', 'wall_ms')} if samples else None
    print('LIVE_BACKWARD_REPLAY_DONE', family, flush=True)
    return record
