import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import flag_gems
from flag_gems.runtime import device as runtime_device


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU attention")
@pytest.mark.parametrize("tf32", [False, True])
@pytest.mark.parametrize(
    "shape,columns,value_dim,layout,mask_kind,scale,causal",
    [
        ((2, 7, 16), 19, 16, "contiguous", "none", 1.0, False),
        ((3, 23, 32), 65, 32, "transpose", "float", 1.0, False),
        ((2, 4, 32, 32), 128, 32, "slice", "float", 1.0, False),
        ((1, 3, 47, 64), 257, 48, "contiguous", "float", None, False),
        ((2, 2, 17, 32), 63, 16, "transpose", "bool", 0.5, False),
        ((1, 2, 31, 16), 67, 32, "slice", "masked_row", 1.0, False),
        ((2, 19, 32), 37, 16, "contiguous", "none", None, True),
        ((1, 2, 29, 32), 43, 32, "transpose", "float", -0.5, False),
        ((1, 2, 13, 16), 47, 16, "contiguous", "float", 0.0, False),
    ],
)
def test_ppu_sdpa_preserves_math_path(shape, columns, value_dim, layout, mask_kind, scale, causal, tf32):
    generator = torch.Generator().manual_seed(8040)

    def tensor(s):
        if layout == "transpose":
            return torch.randn((*s[:-2], s[-1], s[-2]), generator=generator).to(flag_gems.device).transpose(-1, -2)
        if layout == "slice":
            return torch.randn((*s[:-1], 2 * s[-1] + 1), generator=generator).to(flag_gems.device)[..., 1::2]
        return torch.randn(s, generator=generator).to(flag_gems.device)

    q = tensor(shape)
    k = tensor((*shape[:-2], columns, shape[-1]))
    v = tensor((*shape[:-2], columns, value_dim))
    mask = None
    if mask_kind != "none":
        mask = torch.randn((shape[-2], columns), generator=generator).to(flag_gems.device)
        if mask_kind == "bool":
            mask = mask > 0
        elif mask_kind == "masked_row":
            mask[0] = -float("inf")
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), flag_gems.use_gems(exclude=["_scaled_dot_product_attention_math"]):
            expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, is_causal=causal)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), flag_gems.use_gems():
            actual = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, is_causal=causal)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        cpu = F.scaled_dot_product_attention(q.cpu().double(), k.cpu().double(), v.cpu().double(),
                                            attn_mask=None if mask is None else mask.cpu().double() if mask.dtype != torch.bool else mask.cpu(),
                                            scale=scale, is_causal=causal)
        torch.testing.assert_close(actual.cpu(), cpu.float(), atol=0.008 if tf32 else 1e-5, rtol=0.008 if tf32 else 1e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.skipif(runtime_device.vendor_name != "thead", reason="PPU attention")
def test_ppu_sdpa_training_retains_autograd():
    generator = torch.Generator().manual_seed(8041)
    cpu = [torch.randn(s, generator=generator).double().requires_grad_(True)
           for s in [(1, 2, 7, 32), (1, 2, 19, 32), (1, 2, 19, 32), (7, 19)]]
    actual_inputs = [x.detach().float().to(flag_gems.device).requires_grad_(True) for x in cpu]
    expected = F.scaled_dot_product_attention(*cpu[:3], attn_mask=cpu[3], scale=1.0)
    expected.square().sum().backward()
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with sdpa_kernel(SDPBackend.MATH), flag_gems.use_gems():
            actual = F.scaled_dot_product_attention(*actual_inputs[:3], attn_mask=actual_inputs[3], scale=1.0)
            actual.square().sum().backward()
        torch.testing.assert_close(actual.cpu(), expected.float(), atol=1e-5, rtol=1e-5)
        for tensor, reference in zip(actual_inputs, cpu):
            assert tensor.grad is not None
            torch.testing.assert_close(tensor.grad.cpu(), reference.grad.float(), atol=2e-4, rtol=2e-4)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
