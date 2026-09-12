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


@pytest.mark.div_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.true_divide(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.true_divide(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_inp1.true_divide_(ref_inp2)
    with flag_gems.use_gems():
        inp1.true_divide_(inp2)

    utils.gems_assert_close(inp1, ref_inp1, dtype, equal_nan=True)


@pytest.mark.div_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_out(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.empty_like(ref_inp1)
    res_out = torch.empty_like(inp1)

    torch.true_divide(ref_inp1, ref_inp2, out=ref_out)
    with flag_gems.use_gems():
        torch.true_divide(inp1, inp2, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1, False)

    ref_out = torch.true_divide(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.true_divide(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_scalar_(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1.clone(), False)

    ref_inp1.true_divide_(inp2)
    with flag_gems.use_gems():
        inp1.true_divide_(inp2)

    utils.gems_assert_close(inp1, ref_inp1, dtype, equal_nan=True)


@pytest.mark.div_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.true_divide(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.true_divide(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("tensor_divisor", [False, True])
def test_true_divide_inplace_autograd(tensor_divisor):
    source_ref = torch.randn(
        (4, 3), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    source = source_ref.detach().clone().requires_grad_(True)
    if tensor_divisor:
        divisor_ref = (
            torch.rand((1, 3), dtype=torch.float32, device=flag_gems.device) + 0.5
        ).requires_grad_(True)
        divisor = divisor_ref.detach().clone().requires_grad_(True)
    else:
        divisor_ref = divisor = 2.5

    ref_out = source_ref.clone()
    ref_out.true_divide_(divisor_ref)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)

    with flag_gems.use_gems(exclude=[]):
        out = source.clone()
        out.true_divide_(divisor)
        out.backward(grad_out)

    utils.gems_assert_close(out, ref_out, torch.float32)
    utils.gems_assert_close(source.grad, source_ref.grad, torch.float32)
    if tensor_divisor:
        utils.gems_assert_close(divisor.grad, divisor_ref.grad, torch.float32)


@pytest.mark.true_divide
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_dispatch(shape, dtype, caplog):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.ops.aten.true_divide.Tensor(ref_inp1, ref_inp2)
    with caplog.at_level("DEBUG", logger="flag_gems.ops.true_divide"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.true_divide.Tensor(inp1, inp2)

    assert "GEMS TRUE_DIVIDE" in caplog.text
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
