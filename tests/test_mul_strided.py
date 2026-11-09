"""Broadcast multiplication must preserve output views and dtype semantics."""
import pytest
import torch

import flag_gems


@pytest.mark.mul
@pytest.mark.parametrize("rank", [1, 3, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64, torch.bool])
def test_mul_strided_output_matches_cpu(rank, dtype):
    generator = torch.Generator().manual_seed(4081)
    shape = (3,) * (rank - 1) + (17,)
    low, high = (0, 2) if dtype == torch.bool else (-9, 10)
    a = torch.randint(low, high, shape, generator=generator).to(dtype)
    b = torch.randint(low, high, (17,), generator=generator).to(dtype)
    if rank > 1:
        a = a.transpose(0, 1)
    expected = torch.mul(a, b)
    a_device, b_device = a.to(flag_gems.device), b.to(flag_gems.device)
    backing = torch.zeros(shape[:-1] + (34,), dtype=dtype, device=flag_gems.device)
    output = backing[..., 1::2]
    with flag_gems.use_gems():
        actual = torch.mul(a_device, b_device, out=output)
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert torch.count_nonzero(backing[..., ::2]).item() == 0
