# Copyright 2026 FlagOS Contributors
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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIMS_LIST = [1]
    KIND_KEEPDIM_DIMS_SHAPE = [("normal", True, 1, utils.REDUCTION_SHAPES[0])]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    KIND_KEEPDIM_DIMS_SHAPE = list(
        zip(
            ["normal", "allTrue"] * 2,
            [True, False] * 2,
            DIMS_LIST,
            utils.REDUCTION_SHAPES + [(7, 4, 11, 1)],
        )
    )


@pytest.mark.any
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_any(shape, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.any(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.any(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.any_dims
@pytest.mark.parametrize("kind, keepdim, dim, shape", KIND_KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
def test_any_dims(shape, dim, keepdim, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.any(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.any(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.any_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("kind", ["normal", "allFalse", "allTrue"])
def test_any_dim(shape, dtype, keepdim, dim, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    elif kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.any(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.any(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.any
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bool])
def test_any_empty(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.any(utils.to_reference(inp))

    with flag_gems.use_gems():
        res_out = torch.any(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.any_dims
@pytest.mark.parametrize(
    "dim, keepdim",
    [(0, False), (1, False), (1, True), ([0, 1], False), ([0, 1], True)],
)
def test_any_dims_empty(dim, keepdim):
    inp = torch.empty((2, 0, 3), dtype=torch.float32, device=flag_gems.device)
    ref_out = torch.any(utils.to_reference(inp), dim=dim, keepdim=keepdim)

    with flag_gems.use_gems():
        res_out = torch.any(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)
