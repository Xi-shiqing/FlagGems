"""Cache immutable metadata for FlagGems' FP32 generic broadcast Mul."""
from functools import lru_cache
import hashlib
import importlib
import inspect
from textwrap import dedent

import torch


class MulMetadata:
    def __init__(self):
        module = importlib.import_module("flag_gems.ops.mul")
        original = dedent(inspect.getsource(module._launch_generic))
        expected = (
            "torch.tensor(\n"
            "            shape + a_stride + b_stride + out_stride,\n"
            "            device=output.device,\n"
            "            dtype=torch.int64,\n"
            "        )"
        )
        assert original.count(expected) == 1
        patched = original.replace(
            expected, "_cached_mul_meta(shape + a_stride + b_stride + out_stride, output.device)"
        )

        @lru_cache(maxsize=512)
        def immutable(values, device, stream):
            if torch.musa.is_current_stream_capturing():
                raise RuntimeError("Unseen FlagGems Mul metadata while capturing")
            return torch.tensor(values, device=device, dtype=torch.int64)

        def metadata(values, device):
            return immutable(values, device, torch.musa.current_stream(device).musa_stream)

        self.cache = immutable
        self.patch = {"file": inspect.getsourcefile(module._launch_generic),
                      "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                      "patched_source": patched}
        namespace = dict(module.__dict__, _cached_mul_meta=metadata)
        exec(compile(patched, "<FlagGems:mul_metadata>", "exec"), namespace)
        module._launch_generic = namespace["_launch_generic"]

    def stats(self):
        return self.cache.cache_info()._asdict()
