"""Shared output allocation must honor both opt-in controls and grad mode."""
import pytest
import torch

import flag_gems
from flag_gems.utils import full_write_allocation as bridge
from flag_gems.utils import native_output_allocation as allocator

pytestmark = pytest.mark.skipif(
    not allocator.ENABLED,
    reason='requires FLAG_GEMS_NATIVE_FULL_WRITE=1 and its built extension',
)


@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'slice', 'empty', 'scalar'])
def test_native_flag_reaches_legacy_helper(monkeypatch, layout):
    x = torch.empty((7, 19), device=flag_gems.device)
    if layout == 'transpose':
        x = x.t()
    elif layout == 'slice':
        x = x[:, ::2]
    elif layout == 'empty':
        x = x[:0]
    elif layout == 'scalar':
        x = x[0, 0]
    monkeypatch.setattr(bridge, 'ENABLED', False)
    native = allocator._empty
    calls = []

    def observed(*args):
        output = native(*args)
        calls.append(output)
        return output

    monkeypatch.setattr(allocator, '_empty', observed)
    with torch.no_grad():
        output = bridge.empty_like(x)
    assert len(calls) == (0 if layout == 'slice' else 1)
    assert output.shape == x.shape and output.dtype == x.dtype
    assert output.stride() == torch.empty_like(x).stride()
    if x.numel():
        assert output.data_ptr() != x.data_ptr()
    with torch.enable_grad():
        bridge.empty_like(x)
        bridge.empty((3, 7), dtype=x.dtype, device=x.device)
    assert len(calls) == (0 if layout == 'slice' else 1)


def test_both_controls_disabled_keep_ordinary_allocation(monkeypatch):
    monkeypatch.setattr(bridge, 'ENABLED', False)
    monkeypatch.setattr(allocator, 'ENABLED', False)

    def forbidden(*args):
        raise AssertionError('disabled optional allocator was used')

    monkeypatch.setattr(allocator, '_empty', forbidden)
    x = torch.empty((7, 19), device=flag_gems.device)
    with torch.no_grad():
        assert bridge.empty_like(x).shape == x.shape
        assert bridge.empty((3, 7), dtype=x.dtype, device=x.device).shape == (3, 7)
