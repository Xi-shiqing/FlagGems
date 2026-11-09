"""Keep strides and storage independence when a redundant view is omitted."""
import pytest
import torch

import flag_gems
from flag_gems.utils import full_write_allocation as bridge
from flag_gems.utils import native_output_allocation as allocator

pytestmark = pytest.mark.skipif(not allocator.ENABLED, reason='requires the optional allocator')


@pytest.mark.parametrize('shape,stride', [((1, 19), (99, 1)), ((7, 1, 19), (19, 211, 1)),
                                         ((0, 19), (73, 1)), ((19, 7), (1, 19)),
                                         ((7, 19), (19, 1)), ((), ())])
def test_output_stride_and_fresh_storage(monkeypatch, shape, stride):
    x = torch.empty_strided(shape, stride, device=flag_gems.device)
    monkeypatch.setattr(bridge, 'ENABLED', False)
    with torch.no_grad():
        out = bridge.empty_like(x)
    assert out.shape == x.shape and out.stride() == x.stride()
    if x.numel():
        assert out.data_ptr() != x.data_ptr()
    if out.stride() == torch.empty(shape, device=x.device).stride():
        assert out._base is None
