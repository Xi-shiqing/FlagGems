"""Candidate replays of live operands; teacher prediction is returned unchanged."""
import importlib
import inspect
import os
import statistics
from functools import wraps
from audit_routes import metrics, tensor_meta

os.environ.pop('PROTENIX_TF32_CAPTURE_DIR', None)


class TriangleReplay:
    def __init__(self, model):
        self.ppu = os.environ.get('PROTENIX_TRIATTENTION_FLAGGEMS_PUBLIC') == '1'
        self.namespace = importlib.import_module('flag_gems.ops' if self.ppu else 'cuequivariance_ops_torch')
        self.original = self.namespace.triangle_attention
        self.signature = inspect.signature(self.original)
        self.parameters = list(self.signature.parameters)[:3]
        self.candidates = {name: importlib.import_module(('' if self.ppu else 'cuda_') + name)
                           for name in ('forward_kv_prequant',)}
        self.mode = self.family = None
        self.rows, self.seen, self.forwards, self.selected = {}, set(), [], []
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
        previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        if self.ppu:
            q, k, v = operands
            bias1, bias2 = (bound.arguments[key] for key in ('bias1', 'bias2'))
            scale = 1.
            canonical_teacher = teacher
        else:
            q, k, v = [value.unsqueeze(0) if value.ndim == 4 else value for value in operands]
            q, k, v = [value.permute(0, 1, 3, 2, 4).contiguous() for value in (q, k, v)]
            bias2 = bound.arguments['bias']
            if bias2.ndim == 4:
                bias2 = bias2.unsqueeze(0)
            mask = bound.arguments.get('mask')
            if mask is None:
                bias1 = torch.zeros((q.shape[0], q.shape[1], 1, 1, q.shape[2]), device=q.device)
            else:
                if mask.ndim == 4:
                    mask = mask.unsqueeze(0)
                bias1 = -1e9 * (~mask).float()
            scale = bound.arguments.get('scale')
            if scale is None:
                scale = q.shape[-1]**-.5
            value = teacher.unsqueeze(0) if teacher.ndim == 4 else teacher
            canonical_teacher = value.permute(0, 1, 3, 2, 4)
        assert canonical_teacher.shape == q.shape
        groups, queries = sorted(set((0, q.shape[1]-1))), sorted(set((0, q.shape[2]//2, q.shape[2]-1)))
        # FP64 sample retains every key and full reduction dimension for 2 groups x 3 queries.
        a = q[:, groups][:, :, queries].detach().cpu().double().permute(0, 1, 3, 2, 4)
        b, c = [value[:, groups].detach().cpu().double().permute(0, 1, 3, 2, 4) for value in (k, v)]
        bb = bias2.detach().cpu().double()[..., queries, :]
        bm = bias1.detach().cpu().double()
        if bm.shape[1] != 1:
            bm = bm[:, groups]
        gold = ((a @ b.transpose(-1, -2)) * scale + bb + bm).softmax(-1) @ c
        gold = gold.permute(0, 1, 3, 2, 4)
        record = {'family': self.family[0], 'module': self.family[1], 'inputs': [tensor_meta(value) for value in operands],
                  'teacher_output': tensor_meta(teacher), 'canonical_shape': list(q.shape),
                  'sample_gold': {'groups': groups, 'queries': queries, 'full_keys_and_reduction': True, 'dtype': 'CPU FP64'},
                  'scale': scale, 'candidate_q_scale_placement': 'FP32 Q*scale before once-rounding; native CUE uses QK*scale after dot. PPU inputs already scaled.', 'replays': {}}
        values = {'tf32': canonical_teacher}
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            changed = self.signature.bind(*args, **kwargs)
            if self.ppu:
                changed.arguments['precision'] = 'ieee'
            value = self.original(*changed.args, **changed.kwargs)
            if not self.ppu:
                value = value.unsqueeze(0) if value.ndim == 4 else value
                value = value.permute(0, 1, 3, 2, 4)
            values['fp32'] = value
            scaled_q = q if scale == 1. else q * scale
            for name, module in self.candidates.items():
                values[name] = module.triangle_attention(scaled_q, k, v, bias1, bias2, precision='tf32', layout='BNSHD')
            for name, value in values.items():
                record['replays'][name] = {'output_sha256': tensor_meta(value)['sha256'],
                    'vs_unmodified_tf32': metrics(value.detach().cpu(), canonical_teacher.detach().cpu()),
                    'vs_unmodified_fp32': metrics(value.detach().cpu(), values['fp32'].detach().cpu()),
                    'sample_vs_fp64': metrics(value[:, groups][:, :, queries].detach().cpu(), gold)}
            del values
            from live_backward import audit_live_training
            from runtime_precision import suspend_prediction_audit
            with suspend_prediction_audit():
                record["live_backward"] = audit_live_training(self, bound, q, k, v, bias1, bias2, scale)
            record["live_backward"]["timing_instrumentation"] = "This harness's Python profiler and ATen trace suspended; FlagGems library registrations remain active. Teacher model execution audited normally."
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
        self.rows[self.mode].append(record)
        print('LIVE_QUERY_ONCE_REPLAY', self.family, flush=True)
        return teacher

    def close(self):
        assert self.rows.keys() == {'fp32', 'tf32'} and self.rows['tf32']
        return {'status': 'PASS', 'server': 'ppu' if self.ppu else 'h100', 'records': self.rows['tf32'],
                'scope': 'First live call per semantic family, entire candidate output compared to local baselines. CPU FP64 samples retain full keys. Teacher model predictions unchanged; all tensors stay on original server. Custom H100 candidates are not native baselines.'}
