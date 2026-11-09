# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    # Single tiny shape keeps QUICK_MODE fast; mirrors sibling lift_fresh_copy.
    LIFT_FRESH_SHAPES = [(2, 3)]
else:
    # Small-to-medium 2D shapes cover tail masking and multiple blocks; mirrors
    # the sibling lift_fresh_copy test shapes.
    LIFT_FRESH_SHAPES = [(2, 3), (128, 256), (512, 512)]


@pytest.mark.lift_fresh
@pytest.mark.parametrize("shape", LIFT_FRESH_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lift_fresh(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.lift_fresh(ref_inp)
    res_out = flag_gems.lift_fresh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lift_fresh
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.bfloat16, torch.int64, torch.bool, torch.complex64],
)
@pytest.mark.parametrize(
    "layout",
    ["contiguous", "transpose", "offset", "expanded", "empty", "scalar", "channels_last"],
)
def test_lift_fresh_alias_contract(dtype, layout):
    base = torch.arange(48, device=flag_gems.device).to(dtype).reshape(6, 8)
    x = {
        "contiguous": lambda: base,
        "transpose": lambda: base.T,
        "offset": lambda: base[1:, 1::2],
        "expanded": lambda: base[:1].expand(9, 8),
        "empty": lambda: base[:0],
        "scalar": lambda: base[1, 1],
        "channels_last": lambda: base.reshape(1, 6, 1, 8).contiguous(
            memory_format=torch.channels_last
        ),
    }[layout]()
    native = torch.ops.aten.lift_fresh.default(x)
    direct = flag_gems.lift_fresh(self=x)
    with flag_gems.use_gems():
        registered = torch.ops.aten.lift_fresh.default(x)
    for result in [native, direct, registered]:
        assert result is x
        assert result.stride() == x.stride()
        assert result.storage_offset() == x.storage_offset()
        assert result.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()


@pytest.mark.lift_fresh
def test_lift_fresh_construction_and_backward():
    gradients = []
    for use_gems in [False, True]:
        with flag_gems.use_gems() if use_gems else nullcontext():
            x = torch.tensor(
                [0.25, -2.0, 3.0], device=flag_gems.device, requires_grad=True
            )
            # Tensor construction invokes lift_fresh before enabling gradients.
            # Calling lift_fresh a second time on an existing grad-enabled leaf
            # does not represent that operation's freshness precondition.
            assert x.is_leaf and x.requires_grad and x.grad_fn is None
            loss = x.square().sum()
        gradients.append(torch.autograd.grad(loss, x)[0].cpu())
    assert torch.equal(gradients[0], gradients[1])
    assert torch.equal(gradients[0], torch.tensor([0.5, -4.0, 6.0]))
