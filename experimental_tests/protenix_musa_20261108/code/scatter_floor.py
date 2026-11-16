"""Floor local scatter counts without constructing a data-dependent index list."""
import hashlib
import inspect
from textwrap import dedent

import torch


class ScatterFloor:
    def __init__(self, trace):
        from protenix.utils import scatter_utils

        self.op = "flaggems_protenix_scatter::floor_counts_"
        self.definition = torch.library.Library("flaggems_protenix_scatter", "DEF")
        self.definition.define("floor_counts_(Tensor(a!) count) -> Tensor(a!)")
        self.implementation = torch.library.Library("flaggems_protenix_scatter", "IMPL")
        self.implementation.impl(
            "floor_counts_",
            lambda count: trace(self.op, torch.clamp_min_, (count, 1), {}),
            "MUSA",
        )
        original = scatter_utils.scatter_mean
        source = dedent(inspect.getsource(original))
        expected = "count[count < 1] = 1"
        assert source.count(expected) == 1
        patched = source.replace(expected, "torch.ops.flaggems_protenix_scatter.floor_counts_(count)")
        self.patch = {"file": inspect.getsourcefile(original),
                      "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
                      "patched_source": patched}
        namespace = dict(original.__globals__)
        exec(compile(patched, "<7bca-scatter:floor>", "exec"), namespace)
        scatter_utils.scatter_mean = namespace["scatter_mean"]
