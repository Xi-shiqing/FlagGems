import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triangle Attention requires a CUDA device"
)


def _inputs(dtype: torch.dtype, d: int, *, noncontiguous: bool = False):
    torch.manual_seed(20260812 + d)
    shape = (1, 2, 17, 2, d)

    def make(scale: float) -> torch.Tensor:
        width = d * (2 if noncontiguous else 1)
        value = torch.randn((*shape[:-1], width), device="cuda", dtype=dtype) * scale
        return value[..., ::2] if noncontiguous else value

    q, k, v = make(0.15), make(0.15), make(0.20)
    bias1_width = 34 if noncontiguous else 17
    bias1 = torch.zeros((1, 2, 1, 1, bias1_width), device="cuda")
    if noncontiguous:
        bias1 = bias1[..., ::2]
    bias1[..., -2:] = -1.0e4
    bias2_width = 34 if noncontiguous else 17
    bias2 = torch.randn((1, 1, 2, 17, bias2_width), device="cuda") * 0.05
    if noncontiguous:
        bias2 = bias2[..., ::2]
    upstream = torch.randn_like(q) * 0.10
    return q, k, v, bias1, bias2, upstream


def _reference(q, k, v, bias1, bias2, upstream):
    inputs = [
        value.detach().cpu().double().requires_grad_(True) for value in (q, k, v)
    ]
    q64, k64, v64 = inputs
    bias264 = bias2.detach().cpu().double().requires_grad_(True)
    scores = torch.einsum("bnqhd,bnkhd->bnhqk", q64, k64)
    probability = torch.softmax(scores + bias1.cpu().double() + bias264, dim=-1)
    output = torch.einsum("bnhqk,bnkhd->bnqhd", probability, v64)
    grads = torch.autograd.grad(output, (*inputs, bias264), upstream.cpu().double())
    return output, grads


@pytest.mark.parametrize(
    ("dtype", "d", "atol"),
    (
        (torch.float32, 16, 1.0e-5),
        (torch.float32, 32, 1.0e-5),
        (torch.float32, 64, 1.0e-5),
        (torch.float16, 16, 3.0e-3),
        (torch.float16, 32, 3.0e-3),
        (torch.float16, 64, 3.0e-3),
        (torch.bfloat16, 16, 6.0e-3),
        (torch.bfloat16, 32, 6.0e-3),
        (torch.bfloat16, 64, 6.0e-3),
    ),
)
def test_forward_backward_matches_fp64_reference(dtype, d, atol):
    from flag_gems.ops.triangle_attention import triangle_attention

    values = _inputs(dtype, d, noncontiguous=d == 32)
    q0, k0, v0, bias1, bias20, upstream = values
    if d == 16:
        bias1 = bias1.to(dtype)
        bias20 = bias20.to(dtype)
        values = q0, k0, v0, bias1, bias20, upstream
    reference_out, reference_grads = _reference(*values)
    q, k, v = [value.detach().requires_grad_(True) for value in (q0, k0, v0)]
    bias2 = bias20.detach().requires_grad_(True)
    output = triangle_attention(q, k, v, bias1, bias2)
    grads = torch.autograd.grad(output, (q, k, v, bias2), upstream)

    torch.testing.assert_close(output.cpu().double(), reference_out, atol=atol, rtol=0)
    for got, expected in zip(grads, reference_grads, strict=True):
        torch.testing.assert_close(got.cpu().double(), expected, atol=atol, rtol=0)


@pytest.mark.parametrize(
    "precision",
    ("tf32", "qk_tf32", "pv_tf32", "tf32x3", "qk_tf32x3", "pv_tf32x3"),
)
def test_precision_modes_support_first_order_backward(precision):
    from flag_gems.ops.triangle_attention import triangle_attention

    q0, k0, v0, bias1, bias20, upstream = _inputs(torch.float32, 32)
    q, k, v = [value.detach().requires_grad_(True) for value in (q0, k0, v0)]
    bias2 = bias20.detach().requires_grad_(True)
    output = triangle_attention(q, k, v, bias1, bias2, precision=precision)
    grads = torch.autograd.grad(output, (q, k, v, bias2), upstream)
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_bnhsd_layout_forward_and_backward_match_bnshd():
    from flag_gems.ops.triangle_attention import triangle_attention

    q0, k0, v0, bias1, bias20, upstream = _inputs(torch.float32, 32)
    q, k, v = [value.detach().requires_grad_(True) for value in (q0, k0, v0)]
    bias2 = bias20.detach().requires_grad_(True)
    expected = triangle_attention(q, k, v, bias1, bias2)
    expected_grads = torch.autograd.grad(expected, (q, k, v, bias2), upstream)
    qh, kh, vh = [
        value.detach().permute(0, 1, 3, 2, 4).requires_grad_(True)
        for value in (q0, k0, v0)
    ]
    bias2h = bias20.detach().requires_grad_(True)
    actual = triangle_attention(
        qh,
        kh,
        vh,
        bias1,
        bias2h,
        layout="BNHSD",
    )
    torch.testing.assert_close(actual.permute(0, 1, 3, 2, 4), expected)
    actual_grads = torch.autograd.grad(
        actual,
        (qh, kh, vh, bias2h),
        upstream.permute(0, 1, 3, 2, 4),
    )
    for index, (got, wanted) in enumerate(
        zip(actual_grads, expected_grads, strict=True)
    ):
        if index < 3:
            got = got.permute(0, 1, 3, 2, 4)
        torch.testing.assert_close(got, wanted)


def test_bias1_is_explicitly_non_learnable():
    from flag_gems.ops.triangle_attention import triangle_attention

    q, k, v, bias1, bias2, _ = _inputs(torch.float32, 16)
    bias1.requires_grad_(True)
    with pytest.raises(RuntimeError, match="additive mask"):
        triangle_attention(q, k, v, bias1, bias2)


def test_unsupported_input_fails_without_fallback():
    from flag_gems.ops.triangle_attention import triangle_attention

    q = torch.zeros((1, 1, 3, 1, 128), device="cuda")
    bias1 = torch.zeros((1, 1, 1, 1, 3), device="cuda")
    bias2 = torch.zeros((1, 1, 1, 3, 3), device="cuda")
    with pytest.raises(ValueError, match="16/32/64"):
        triangle_attention(q, q, q, bias1, bias2)


def test_real_optimizer_step_updates_all_learnable_inputs():
    from flag_gems.ops.triangle_attention import triangle_attention

    q, k, v, bias1, bias2, _ = _inputs(torch.float32, 32)
    parameters = [torch.nn.Parameter(value) for value in (q, k, v, bias2)]
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.Adam(parameters, lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    output = triangle_attention(*parameters[:3], bias1, parameters[3])
    loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert all(torch.count_nonzero(parameter.grad) > 0 for parameter in parameters)
    optimizer.step()
    assert all(not torch.equal(old, new) for old, new in zip(before, parameters))


def test_module_wrapper_and_public_export():
    from flag_gems import triangle_attention as root_triangle_attention
    from flag_gems.ops import (
        TriangleAttention,
        triangle_attention,
        triangle_attention_support_error,
    )

    q, k, v, bias1, bias2, _ = _inputs(torch.float32, 16)
    expected = triangle_attention(q, k, v, bias1, bias2)
    actual = TriangleAttention()(q, k, v, bias1, bias2)
    torch.testing.assert_close(actual, expected)
    assert root_triangle_attention is triangle_attention
    assert triangle_attention_support_error(q, k, v, bias1, bias2) is None


def test_precision_is_explicit_and_validated(monkeypatch):
    from flag_gems.ops.triangle_attention import triangle_attention

    q, k, v, bias1, bias2, _ = _inputs(torch.float32, 16)
    monkeypatch.setenv("FLAG_GEMS_TRIANGLE_ATTENTION_TF32_MODE", "full")
    default = triangle_attention(q, k, v, bias1, bias2)
    monkeypatch.setenv("FLAG_GEMS_TRIANGLE_ATTENTION_TF32_MODE", "not-a-mode")
    repeated = triangle_attention(q, k, v, bias1, bias2)
    torch.testing.assert_close(repeated, default)
    with pytest.raises(ValueError, match="precision must be one of"):
        triangle_attention(q, k, v, bias1, bias2, precision="automatic")
