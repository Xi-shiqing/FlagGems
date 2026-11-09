"""Registered PPU fill: exact values, views, empty tensors and bounded NaN cache."""
import sys

import pytest
import torch

import flag_gems


SHAPES=[(),(0,),(0,3),(257,),(3,129),(2,17,33),(2,3,5,7),(2,3,5,7,11)]
VALUES=[0,1,-3.25,-0.0,float('inf'),float('-inf'),float('nan'),2**53]


def _bits(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8)


@pytest.mark.fill_scalar_
@pytest.mark.skipif(flag_gems.vendor_name!='thead',reason='PPU scalar fill')
@pytest.mark.parametrize('shape',SHAPES)
@pytest.mark.parametrize('value',VALUES)
def test_ppu_fill_registered_values(shape,value,monkeypatch):
    assert flag_gems.fill_scalar_.__module__.endswith('_thead.ops.fill')
    outputs=[]
    reference=torch.empty(shape,dtype=torch.float32).fill_(value)
    for enabled in ['0','1']:
        monkeypatch.setenv('FLAG_GEMS_PPU_CONTIGUOUS_FILL',enabled)
        x=torch.empty(shape,device=flag_gems.device,dtype=torch.float32)
        with flag_gems.use_gems():
            result=x.fill_(value)
        assert result is x
        actual=result.cpu()
        if isinstance(value,float) and value!=value:
            assert torch.isnan(actual).all()
        else:
            assert torch.equal(_bits(actual),_bits(reference))
        outputs.append(actual)
    assert torch.equal(_bits(outputs[0]),_bits(outputs[1]))


@pytest.mark.fill_scalar_
@pytest.mark.skipif(flag_gems.vendor_name!='thead',reason='PPU scalar fill')
@pytest.mark.parametrize('layout',['slice','transpose','contiguous_offset','rank5','expanded'])
@pytest.mark.parametrize('value',[-7.25,float('nan')])
def test_ppu_fill_preserves_storage_and_views(layout,value,monkeypatch):
    cpu=torch.arange(12*17,dtype=torch.float32).reshape(12,17)
    def view(x):
        if layout=='slice':return x[2:9,3:11]
        if layout=='transpose':return x.t()
        if layout=='contiguous_offset':return x[2:9]
        if layout=='rank5':return x.reshape(1,3,4,1,17).transpose(1,2)
        return x[:1,:1].expand(3,5)
    reference=cpu.clone()
    view(reference).fill_(value)
    x=cpu.to(flag_gems.device)
    target=view(x)
    shape,stride,offset=target.shape,target.stride(),target.storage_offset()
    monkeypatch.setenv('FLAG_GEMS_PPU_CONTIGUOUS_FILL','1')
    with flag_gems.use_gems():
        result=target.fill_(value)
    assert result is target
    assert (result.shape,result.stride(),result.storage_offset())==(shape,stride,offset)
    actual=x.cpu()
    assert torch.equal(torch.isnan(actual),torch.isnan(reference))
    valid=~torch.isnan(reference)
    assert torch.equal(_bits(actual[valid]),_bits(reference[valid]))


@pytest.mark.fill_scalar_
@pytest.mark.skipif(flag_gems.vendor_name!='thead',reason='PPU scalar fill')
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.int32,torch.int64,torch.bool])
def test_ppu_fill_other_dtypes_keep_values(dtype,monkeypatch):
    cpu=torch.ones((3,17),dtype=dtype)
    x=cpu.to(flag_gems.device)
    monkeypatch.setenv('FLAG_GEMS_PPU_CONTIGUOUS_FILL','1')
    with flag_gems.use_gems():x.fill_(0)
    assert torch.equal(_bits(x.cpu()),_bits(cpu.fill_(0)))


@pytest.mark.fill_scalar_
@pytest.mark.skipif(flag_gems.vendor_name!='thead',reason='PPU scalar fill')
@pytest.mark.parametrize('shape',[(0,),(0,3),(3,17)])
def test_ppu_fill_fp32_overflow_matches_native(shape,monkeypatch):
    monkeypatch.setenv('FLAG_GEMS_PPU_CONTIGUOUS_FILL','1')
    with pytest.raises(RuntimeError,match='overflow'):
        torch.empty(shape,dtype=torch.float32).fill_(1e100)
    x=torch.empty(shape,device=flag_gems.device,dtype=torch.float32)
    with flag_gems.use_gems(),pytest.raises(RuntimeError,match='overflow'):
        x.fill_(1e100)


@pytest.mark.fill_scalar_
@pytest.mark.skipif(flag_gems.vendor_name!='thead',reason='PPU scalar fill')
def test_ppu_fill_nan_cache_and_deterministic_allocator(monkeypatch):
    module=sys.modules[flag_gems.fill_scalar_.__module__]
    monkeypatch.setenv('FLAG_GEMS_PPU_CONTIGUOUS_FILL','1')
    previous=torch.are_deterministic_algorithms_enabled()
    warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    previous_fill=torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory=True
        x=torch.empty((1024,),device=flag_gems.device,dtype=torch.float32)
        with flag_gems.use_gems():
            x.fill_(0.0)
            count=sum(len(cache) for cache in module._fill_contiguous.kernel_cache)
            assert count>0
            for _ in range(20):x.fill_(float('nan'))
            assert sum(len(cache) for cache in module._fill_contiguous.kernel_cache)==count
            allocated=torch.empty((3,17),device=flag_gems.device,dtype=torch.float32)
            empty=torch.empty((0,),device=flag_gems.device,dtype=torch.float32)
        assert torch.isnan(x.cpu()).all() and torch.isnan(allocated.cpu()).all()
        assert empty.numel()==0
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.utils.deterministic.fill_uninitialized_memory
    finally:
        torch.utils.deterministic.fill_uninitialized_memory=previous_fill
        torch.use_deterministic_algorithms(previous,warn_only=warn_only)


@pytest.mark.fill_scalar_
@pytest.mark.skipif(flag_gems.vendor_name!='thead',reason='PPU scalar fill')
def test_ppu_fill_large_offset_index(monkeypatch):
    monkeypatch.setenv('FLAG_GEMS_PPU_CONTIGUOUS_FILL','1')
    x=torch.empty((2**31+17,),device=flag_gems.device,dtype=torch.float32)
    with flag_gems.use_gems():x.fill_(-7.25)
    assert torch.equal(x[:17].cpu(),torch.full((17,),-7.25))
    assert torch.equal(x[-17:].cpu(),torch.full((17,),-7.25))
