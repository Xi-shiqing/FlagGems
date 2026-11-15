"""MUSA forward adapter for FlagGems TriangleAttention commit 1ed2f470."""
import hashlib
import importlib.util
from pathlib import Path

import torch


class LaiTriangleMusa:
    op = "flaggems_protenix_lai::triangle_attention"
    upstream_sha256 = "e60e132fdd46578e76b32063958c703a182b7ab6386491d489381ab584cdbfca"

    def __init__(self, trace, output_dir, warps=1):
        from protenix.model.triangular import layers

        path = Path(__file__).with_name("赖冠熹TriAttention") / "triangle_attention.py"
        original = path.read_bytes()
        assert hashlib.sha256(original).hexdigest() == self.upstream_sha256
        # Preserve the original kernel, layouts, and strict IEEE precision.
        source = original.decode()
        for old, new in {
            "if not q.is_cuda:": 'if q.device.type != "musa":',
            "torch.cuda.get_device_name(q.device)": "torch.musa.get_device_name(q.device)",
        }.items():
            assert source.count(old) == 1
            source = source.replace(old, new)
        changes = ["MUSA device guard", "MUSA device-name lookup"]
        if warps != 1:
            config = "block_m, block_n, num_warps, num_stages = 32, 32, 1, 1"
            assert source.count(config) == 1 and warps == 4
            source = source.replace(config, config.replace("32, 32, 1, 1", "32, 32, 4, 1"))
            changes.append("MUSA default launch: 32x32 tiles, 4 warps, 1 stage")
        overlay = output_dir / "lai_triangle_attention_musa_overlay.py"
        overlay.write_text(source)
        spec = importlib.util.spec_from_file_location("lai_triangle_attention_musa_overlay", overlay)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        self.info = {
            "commit": "1ed2f470c5105a3ac8bb1cf875fd16f31744f123",
            "original_sha256": self.upstream_sha256,
            "overlay_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
            "changes": changes,
            "precision": "ieee",
        }
        self.definition = torch.library.Library("flaggems_protenix_lai", "DEF")
        self.definition.define(
            "triangle_attention(Tensor q, Tensor k, Tensor v, Tensor mask_bias, Tensor pair_bias) -> Tensor"
        )
        self.implementation = torch.library.Library("flaggems_protenix_lai", "IMPL")

        def forward(q, k, v, mask_bias, pair_bias):
            assert not torch.is_grad_enabled() and q.dtype == torch.float32
            return self.module.triangle_attention(
                q, k, v, mask_bias, pair_bias, layout="BNHSD", precision="ieee"
            )

        self.implementation.impl(
            "triangle_attention", lambda *args: trace(self.op, forward, args, {}), "MUSA"
        )
        def attention(query, key, value, biases):
            assert len(biases) == 2 and query.ndim in (4, 5)
            if query.ndim == 4:
                mask, pair = biases
                # The chunker may expand a shared bias along the group axis.
                assert pair.shape[0] == 1 or pair.stride(0) == 0, "Unshared pair bias"
                return torch.ops.flaggems_protenix_lai.triangle_attention(
                    *(x.unsqueeze(0) for x in (query, key, value, mask, pair[:1]))
                ).squeeze(0)
            return torch.ops.flaggems_protenix_lai.triangle_attention(query, key, value, *biases)

        layers._attention = attention

    def check(self):
        generator = torch.Generator(device="cpu").manual_seed(817)
        checks = []
        for groups, heads, seq in [(3, 2, 17), (8, 4, 234)]:
            q, k, v = (
                torch.randn((1, groups, heads, seq, 32), generator=generator) * 0.2
                for _ in range(3)
            )
            mask = torch.zeros((1, groups, 1, 1, seq))
            mask[..., -2:] = -1e4
            pair = torch.randn((1, 1, heads, seq, 2 * seq), generator=generator)[..., ::2] * 0.05
            scores = q.double() @ k.double().transpose(-1, -2)
            expected = torch.softmax(scores + mask.double() + pair.double(), dim=-1) @ v.double()
            with torch.no_grad():
                actual = torch.ops.flaggems_protenix_lai.triangle_attention(
                    *(x.to("musa") for x in (q, k, v, mask, pair))
                ).cpu().double()
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
            checks.append({"shape": list(q.shape), "max_absolute_error": (actual - expected).abs().max().item()})
        return checks
