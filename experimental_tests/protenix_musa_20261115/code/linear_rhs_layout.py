"""Test reduction-major weights with the existing strict FP32 FlagGems Linear kernel."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
from pathlib import Path

import torch
import triton
from flag_gems.runtime import torch_device_fn


class LinearRhsLayout:
    def __init__(self, minimum_rows=4096, reuse_packed=False, token_layout=False, token_grouped=False):
        from flag_gems.ops.linear import linear_kernel

        self.kernel = linear_kernel
        self.minimum_rows = minimum_rows
        self.reuse_packed = reuse_packed
        self.cache = ContextVar("linear_rhs_prediction", default=None)
        self.in_token = ContextVar("linear_rhs_token_transformer", default=False) if token_layout else None
        self.implementation = "linear_rhs_layout"
        self.grouped_kernel = None
        source = Path(__file__)
        self.info = {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "minimum_rows": minimum_rows,
            "token_transformer_minimum_rows": 234 if token_layout else None,
            "scope": "contiguous MUSA FP32 inference",
            "packing": "within-prediction reuse" if reuse_packed else "per call",
            "kernel": "flag_gems.ops.linear.linear_kernel; same FP32 accumulation",
        }
        if token_grouped:
            from token_linear_kernel import token_linear_kernel

            self.grouped_kernel = token_linear_kernel
            kernel_source = source.with_name("token_linear_kernel.py")
            self.info["token_grouped_source_sha256"] = hashlib.sha256(kernel_source.read_bytes()).hexdigest()

    def scope_token_transformer(self, transformer):
        original = transformer.forward

        @wraps(original)
        def forward(*args, **kwargs):
            token = self.in_token.set(True)
            try:
                return original(*args, **kwargs)
            finally:
                self.in_token.reset(token)

        transformer.forward = forward

    def select(self, original, trace):
        def dispatch(input, weight, bias=None):
            token_projection = self.in_token is not None and self.in_token.get()
            minimum_rows = 234 if token_projection else self.minimum_rows
            if (
                torch.is_grad_enabled()
                or input.device.type != "musa"
                or torch.is_autocast_enabled("musa")
                or input.dtype != torch.float32
                or weight.dtype != torch.float32
                or weight.device != input.device
                or input.ndim < 2
                or weight.ndim != 2
                or input.shape[-1] != weight.shape[1]
                or weight.numel() == 0
                or not input.is_contiguous()
                or not weight.is_contiguous()
                or input.numel() // weight.shape[1] < minimum_rows
                or (bias is not None and (
                    bias.shape != weight.shape[:1] or bias.dtype != input.dtype
                    or bias.device != input.device or not bias.is_contiguous()
                ))
            ):
                return original(input, weight, bias)
            grouped = token_projection and self.grouped_kernel is not None and all(size % 32 == 0 for size in weight.shape)
            if grouped:
                key = "token_linear_grouped"
            else:
                key = "token_linear_rhs_layout" if token_projection else self.implementation
            return trace(key, self.run, (input, weight, bias, grouped), {})

        return dispatch

    @contextmanager
    def prediction(self):
        entries = {}
        stats = {"hits": 0, "packs": 0}
        token = self.cache.set((entries, stats))
        try:
            yield stats
        finally:
            self.cache.reset(token)

    def packed_weight(self, weight):
        state = self.cache.get()
        if state is None or torch.is_inference(weight):
            return weight.transpose(0, 1).contiguous()
        entries, stats = state
        stream = torch.musa.current_stream(weight.device).musa_stream
        key = (id(weight), weight.device, stream)
        version = (weight._version, weight.data_ptr(), tuple(weight.shape), weight.stride())
        entry = entries.get(key)
        if entry is not None and entry[1] == version:
            stats["hits"] += 1
            return entry[2]
        packed = weight.transpose(0, 1).contiguous()
        # Keep the source alive, preventing reuse of its Python identity.
        entries[key] = (weight, version, packed)
        stats["packs"] += 1
        return packed

    def run(self, input, weight, bias, grouped=False):
        n, k = weight.shape
        lhs = input.view(-1, k)
        m = lhs.shape[0]
        rhs = self.packed_weight(weight) if self.reuse_packed else weight.transpose(0, 1).contiguous()
        output = torch.empty((m, n), device=input.device, dtype=input.dtype)

        def grid(meta):
            return triton.cdiv(m, meta["BLOCK_SIZE_M"]), triton.cdiv(n, meta["BLOCK_SIZE_N"])

        def grouped_grid(meta):
            return (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)

        with torch_device_fn.device(input.device):
            bias_or_rhs = bias if bias is not None else rhs
            if grouped:
                self.grouped_kernel[grouped_grid](lhs, rhs, bias_or_rhs, output, m, n, k, BIAS=bias is not None)
            else:
                self.kernel[grid](
                    lhs, rhs, bias_or_rhs, output, m, n, k,
                    lhs.stride(0), lhs.stride(1), rhs.stride(1), rhs.stride(0),
                    output.stride(0), output.stride(1), bias.stride(0) if bias is not None else 0,
                    BIAS=bias is not None,
                )
        return output.view(*input.shape[:-1], n)

    def check(self):
        import numpy as np

        # Frozen synthetic fixtures with odd reduction/output sizes and both bias modes.
        rng = np.random.default_rng(817)
        receipts = []
        with torch.no_grad():
            fixtures = [(4097, 129, 65, True), (27378, 64, 128, False)]
            if self.in_token is not None:
                fixtures += [(1170, 768, 1536, False), (1170, 1536, 768, False),
                             (1170, 768, 768, False), (1170, 384, 768, True)]
            for index, (rows, k, n, with_bias) in enumerate(fixtures):
                lhs = rng.standard_normal((rows, k), dtype=np.float32)
                rhs = rng.standard_normal((n, k), dtype=np.float32)
                bias = rng.standard_normal(n, dtype=np.float32) if with_bias else None
                device_bias = torch.from_numpy(bias).to("musa") if with_bias else None
                device_lhs = torch.from_numpy(lhs).to("musa")
                device_rhs = torch.from_numpy(rhs).to("musa")
                grouped = self.grouped_kernel is not None and index >= 2
                actual = self.run(device_lhs, device_rhs, device_bias, grouped)
                if self.reuse_packed:
                    with self.prediction() as stats:
                        self.run(device_lhs, device_rhs, device_bias, grouped)
                        reused = self.run(device_lhs, device_rhs, device_bias, grouped)
                        assert stats == {"hits": 1, "packs": 1}
                        torch.testing.assert_close(reused, actual, rtol=0, atol=0)
                        device_rhs.add_(0.25)
                        updated = self.run(device_lhs, device_rhs, device_bias, grouped)
                        assert stats == {"hits": 1, "packs": 2}
                    fresh = self.run(device_lhs, device_rhs, device_bias, grouped)
                    torch.testing.assert_close(updated, fresh, rtol=0, atol=0)
                observed = actual.cpu().numpy()
                left, right = lhs.astype(np.float64), rhs.astype(np.float64).T
                expected = left @ right
                magnitude = abs(left) @ abs(right)
                if bias is not None:
                    expected += bias
                    magnitude += abs(bias)
                unit_roundoff = 2**-24
                steps = 2 * k + 2
                bound = (steps * unit_roundoff / (1 - steps * unit_roundoff)) * magnitude
                error = abs(observed - expected)
                assert np.isfinite(observed).all() and np.all(error <= bound), (rows, k, n, error.max())
                receipts.append({
                    "shape_mkn": [rows, k, n], "bias": with_bias, "grouped": grouped,
                    "max_absolute_error": float(error.max()),
                    "max_fraction_of_fp32_roundoff_bound": float((error / bound).max()),
                })
        return receipts
