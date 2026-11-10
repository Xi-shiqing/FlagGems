"""Dense permutations preserve special bits, offsets, gradients and streams."""
import importlib
import itertools

import pytest
import torch

import flag_gems


@pytest.mark.bmm
@pytest.mark.parametrize("permutation",list(itertools.permutations(range(3))))
def test_dense_copy_bits_and_gradients(monkeypatch,permutation):
    module=importlib.import_module("flag_gems.runtime.backend._thead.ops.dense_layout_copy")
    monkeypatch.setattr(module,"ENABLED",True)
    physical=torch.arange(33*65*67,dtype=torch.int32).reshape(33,65,67)
    bits=torch.tensor([0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0xffa12345,1,0x80000001],dtype=torch.int64).to(torch.int32)
    physical.flatten()[:8].copy_(bits)
    host=physical.view(torch.float32).permute(permutation)
    storage=torch.empty(host.numel()+7,device=flag_gems.device)
    device=storage.as_strided(host.shape,host.stride(),7);device.copy_(host);device.requires_grad_()
    spec=module._layout_spec(tuple(host.shape),tuple(host.stride()))
    output=module.try_copy(device)
    if spec is None or spec[0]!="transpose":
        assert output is None
        return
    assert output.is_contiguous() and output.untyped_storage().data_ptr()!=device.untyped_storage().data_ptr()
    assert torch.equal(output.cpu().view(torch.int32),host.contiguous().view(torch.int32))
    gradient=torch.autograd.grad(output.sum(),device)[0]
    assert torch.equal(gradient,torch.ones_like(gradient))
    monkeypatch.setattr(module,"empty_full_write",lambda shape,**kwargs:torch.full(shape,float("nan"),**kwargs))
    with torch.no_grad():
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            copied=module.try_copy(device)
            consumed=copied.view(torch.int32).clone()
        stream.synchronize()
        assert torch.equal(consumed.cpu(),host.contiguous().view(torch.int32))
