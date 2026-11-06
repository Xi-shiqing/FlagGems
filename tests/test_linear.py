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

FWD_SHAPES = [
    (16, 128, 64),
    (1, 128, 64),
    (2, 100, 70),
    (5, 512, 129),
    (6, 512, 129),
    (4, 1024, 512),
    (2, 4096, 64),
    (0, 128, 64),
    (7, 256, 192),
    (48, 1024, 512),
    (192, 1024, 1024),
]


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("batch_size, in_features, out_features", FWD_SHAPES)
def test_linear_2d_with_bias(dtype, batch_size, in_features, out_features):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    input_tensor = torch.randn(
        (batch_size, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    bias = torch.randn((out_features,), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    res_out = flag_gems.linear(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("batch_size, in_features, out_features", FWD_SHAPES)
def test_linear_2d_without_bias(dtype, batch_size, in_features, out_features):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    input_tensor = torch.randn(
        (batch_size, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight)
    res_out = flag_gems.linear(input_tensor, weight)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


FWD_3D_SHAPES = [
    (4, 8, 128, 64),
    (1, 1, 128, 64),
    (2, 2, 100, 70),
    (2, 4, 256, 192),
    (8, 24, 1024, 1024),
]


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("batch1, batch2, in_features, out_features", FWD_3D_SHAPES)
def test_linear_3d_with_bias(dtype, batch1, batch2, in_features, out_features):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    input_tensor = torch.randn(
        (batch1, batch2, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    bias = torch.randn((out_features,), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    res_out = flag_gems.linear(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


FWD_1D_SHAPES = [
    (128, 64),
    (16, 16),
    (129, 70),
]


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("in_features, out_features", FWD_1D_SHAPES)
def test_linear_1d_with_bias(dtype, in_features, out_features):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    input_tensor = torch.randn((in_features,), dtype=dtype, device=flag_gems.device)
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    bias = torch.randn((out_features,), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    res_out = flag_gems.linear(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


@pytest.mark.linear_backward
@pytest.mark.parametrize("layout", ["batch_transpose", "last_dim_stride"])
def test_linear_backward_noncontiguous(layout):
    generator = torch.Generator().manual_seed(20261106)
    cpu_input = torch.randn((5, 3, 17), generator=generator).transpose(0, 1)
    cpu_weight = torch.randn((13, 17), generator=generator)
    if layout == "batch_transpose":
        cpu_grad = torch.randn((5, 3, 13), generator=generator).transpose(0, 1)
    else:
        cpu_grad = torch.randn((3, 5, 26), generator=generator)[..., ::2]
    input_tensor = cpu_input.to(flag_gems.device)
    grad_output = cpu_grad.to(flag_gems.device)
    weight = cpu_weight.to(flag_gems.device)
    assert not input_tensor.is_contiguous()
    # Device copies may pack sliced input; reconstruct its stride explicitly.
    if layout == "last_dim_stride":
        grad_output = torch.empty((3, 5, 26), device=flag_gems.device)[..., ::2].copy_(grad_output)
    assert not grad_output.is_contiguous()
    expected = (
        cpu_grad.double() @ cpu_weight.double(),
        cpu_grad.double().reshape(-1, 13).T @ cpu_input.double().reshape(-1, 17),
        cpu_grad.double().sum((0, 1)),
    )
    prior = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with flag_gems.use_gems(exclude=[]):
            actual = torch.ops.aten.linear_backward.default(
                input_tensor, grad_output, weight, (True, True, True)
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value.cpu().double(), reference, rtol=2e-5, atol=2e-5)


@pytest.mark.linear
@pytest.mark.parametrize("allow_tf32", [False, True])
@pytest.mark.parametrize("case", ["batch_transpose", "empty_width", "empty_batch", "empty_output"])
def test_linear_layout_and_empty_dimensions(allow_tf32, case):
    generator = torch.Generator().manual_seed(20261106)
    shape = (5, 3, 128) if case == "batch_transpose" else (
        (3, 5, 0) if case == "empty_width" else (
            (0, 5, 128) if case == "empty_batch" else (3, 5, 128)
        )
    )
    cpu_input = torch.randn(shape, generator=generator)
    if case == "batch_transpose":
        cpu_input = cpu_input.transpose(0, 1)
    width = cpu_input.shape[-1]
    output_width = 0 if case == "empty_output" else 64
    cpu_weight = torch.randn((output_width, width), generator=generator)
    cpu_bias = torch.randn((output_width,), generator=generator)
    device_values = [value.to(flag_gems.device) for value in (cpu_input, cpu_weight, cpu_bias)]
    if case == "batch_transpose":
        assert not device_values[0].is_contiguous()
    prior = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        expected = torch.nn.functional.linear(cpu_input.double(), cpu_weight.double(), cpu_bias.double())
        if allow_tf32 and flag_gems.vendor_name == "thead":
            def rounded(value):
                bits = value.contiguous().view(torch.int32).to(torch.int64)
                bits = (bits + 4095 + ((bits >> 13) & 1)) & -8192
                return bits.to(torch.int32).view(torch.float32).double()

            expected = torch.nn.functional.linear(rounded(cpu_input), rounded(cpu_weight), cpu_bias.double())
        with flag_gems.use_gems(exclude=[]):
            actual = torch.nn.functional.linear(*device_values)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.linear_backward
@pytest.mark.parametrize("case", ["empty_width", "empty_batch", "empty_output"])
def test_linear_backward_empty_dimensions(case):
    rows = 0 if case == "empty_batch" else 3
    width = 0 if case == "empty_width" else 17
    outputs = 0 if case == "empty_output" else 13
    generator = torch.Generator().manual_seed(20261106)
    cpu_input = torch.randn((rows, 5, width), generator=generator)
    cpu_weight = torch.randn((outputs, width), generator=generator)
    cpu_grad = torch.randn((rows, 5, outputs), generator=generator)
    expected = (
        cpu_grad.double() @ cpu_weight.double(),
        cpu_grad.double().reshape(rows * 5, outputs).T @ cpu_input.double().reshape(rows * 5, width),
        cpu_grad.double().sum((0, 1)),
    )
    device_values = [value.to(flag_gems.device) for value in (cpu_input, cpu_grad, cpu_weight)]
    with flag_gems.use_gems(exclude=[]):
        actual = torch.ops.aten.linear_backward.default(*device_values, (True, True, True))
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value.cpu().double(), reference, rtol=2e-5, atol=2e-5)
