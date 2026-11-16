"""Replay one warmed, fixed-shape diffusion step inside a single prediction."""
from contextlib import contextmanager

import torch

from diffusion_reuse import tensor_key


class DiffusionGraph:
    DYNAMIC = ("r_noisy", "t_hat_noise_level")

    def __init__(self, model, audit_enabled):
        self.module = model.diffusion_module
        self.original = self.module.f_forward
        self.module.f_forward = self.forward
        self.audit_enabled = audit_enabled
        self.active = False
        self.graph = None
        self.receipt = {}

    @staticmethod
    def key(value):
        if isinstance(value, torch.Tensor):
            return tensor_key(value)
        if isinstance(value, dict):
            return tuple((name, DiffusionGraph.key(item)) for name, item in value.items())
        if isinstance(value, (list, tuple)):
            return tuple(DiffusionGraph.key(item) for item in value)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise TypeError(f"Unverified static graph input: {type(value).__name__}")

    @staticmethod
    def signature(value):
        return value.shape, value.stride(), value.dtype, value.device

    def forward(self, **inputs):
        if not self.active:
            return self.original(**inputs)
        assert not torch.is_grad_enabled() and not self.module.training
        keys = {name: self.key(value) for name, value in inputs.items() if name not in self.DYNAMIC}
        if self.graph is None:
            self.capture(inputs, keys)
        else:
            assert keys == self.static_keys, "Fixed diffusion graph inputs changed"
            for name in self.DYNAMIC:
                assert self.dynamic_signatures[name] == self.signature(inputs[name])
                self.static[name].copy_(inputs[name])
        if self.audit_enabled() and self.receipt["replays"] < 2:
            self.audit_replay(inputs)
        else:
            self.graph.replay()
        self.receipt["replays"] += 1
        return self.output.clone()

    def audit_replay(self, inputs):
        expected = self.original(**inputs)
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        self.graph.replay()
        end.record()
        torch.musa.synchronize()
        observed = self.output.detach().cpu()
        reference = expected.detach().cpu()
        assert observed.shape == reference.shape and observed.dtype == reference.dtype
        assert torch.isfinite(observed).all() and torch.isfinite(reference).all()
        elapsed = start.elapsed_time(end)
        assert elapsed > 0
        self.receipt["device_replay_probes"].append({
            "step": self.receipt["replays"] + 1, "elapsed_ms": elapsed,
            "bitwise_equal_to_eager": torch.equal(observed, reference),
            "max_abs_delta_vs_eager": (observed - reference).abs().max().item(),
        })

    def capture(self, inputs, keys):
        self.static = dict(inputs)
        self.static_keys = keys
        self.dynamic_signatures = {name: self.signature(inputs[name]) for name in self.DYNAMIC}
        current = torch.musa.current_stream()
        self.stream = torch.musa.Stream()
        self.stream.wait_stream(current)
        with torch.musa.stream(self.stream):
            for name in self.DYNAMIC:
                self.static[name] = inputs[name].clone(memory_format=torch.preserve_format)
            self.original(**self.static)  # Initialize fixed-shape kernels on a side stream.
        current.wait_stream(self.stream)
        self.graph = torch.musa.MUSAGraph()
        with torch.musa.graph(self.graph, stream=self.stream):
            self.output = self.original(**self.static)
        self.receipt["captures"] += 1

    @contextmanager
    def prediction(self):
        self.receipt = {"captures": 0, "replays": 0, "device_replay_probes": []}
        self.active = True
        try:
            yield
            torch.musa.synchronize()  # Keep graph storage alive until the last replay completes.
        finally:
            self.active = False
            self.graph = None
            self.static = None
            self.output = None
