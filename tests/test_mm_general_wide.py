"""Registered MM and backward keep original bits across a cache-policy switch."""
import importlib

import pytest
import torch

import flag_gems


def backend():
    return importlib.import_module(flag_gems.mm.__globals__["_project_mm"].__module__)


@pytest.mark.mm
@pytest.mark.parametrize("shape", [(1225,768,768),(430,768,384),(245,384,384),(127,139,259),(1985,128,128)])
@pytest.mark.parametrize("transpose_rhs", [False,True])
@pytest.mark.parametrize("tf32", [False,True])
def test_registered_wide_mm(monkeypatch,shape,transpose_rhs,tf32):
    module=backend();m,n,k=shape
    generator=torch.Generator().manual_seed(20261110)
    ah=torch.randn((m,k),generator=generator)
    bh=torch.randn((n,k),generator=generator).t() if transpose_rhs else torch.randn((k,n),generator=generator)
    a=ah.to(flag_gems.device)
    b=torch.empty_strided(bh.shape,bh.stride(),device=flag_gems.device);b.copy_(bh)
    previous=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=tf32
    values=[]
    try:
        for enabled in [False,True,False,True]:
            monkeypatch.setattr(module,"_GENERAL_WIDE_ENABLED",enabled)
            with flag_gems.use_gems(),torch.no_grad():values.append(torch.mm(a,b))
        for value in values[1:]:assert torch.equal(value.view(torch.uint8),values[0].view(torch.uint8))
        if not tf32:
            reference=ah[:16].double()@bh[:,:16].double()
            torch.testing.assert_close(values[0][:16,:16].cpu().double(),reference,rtol=2e-5,atol=3e-5)
        output=torch.empty((n,m),device=a.device).t()
        with flag_gems.use_gems(),torch.no_grad():returned=torch.mm(a,b,out=output)
        assert returned is output
        assert torch.equal(output.contiguous().view(torch.uint8),values[0].view(torch.uint8))
    finally:torch.backends.cuda.matmul.allow_tf32=previous


@pytest.mark.mm
def test_wide_mm_backward(monkeypatch):
    module=backend();generator=torch.Generator().manual_seed(20261110)
    ah=torch.randn((1225,384),generator=generator);bh=torch.randn((384,768),generator=generator)
    previous=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    values=[]
    try:
        for enabled in [False,True]:
            monkeypatch.setattr(module,"_GENERAL_WIDE_ENABLED",enabled)
            a=ah.to(flag_gems.device).requires_grad_();b=bh.to(flag_gems.device).requires_grad_()
            with flag_gems.use_gems():
                output=torch.mm(a,b);gradients=torch.autograd.grad(output.square().sum(),[a,b])
            values.append((output,*gradients))
        for old,new in zip(*values):assert torch.equal(old.contiguous().view(torch.uint8),new.contiguous().view(torch.uint8))
    finally:torch.backends.cuda.matmul.allow_tf32=previous


@pytest.mark.mm
@pytest.mark.parametrize("tf32",[False,True])
def test_wide_compiled_tiles_and_resource_tail(monkeypatch,tf32):
    module=backend()
    sms=module._general_device_sms(torch.device(flag_gems.device))
    assert sms>0
    original=module.mm_kernel_general
    observed=[]
    class Kernel:
        def __getitem__(self,grid):
            launch=original[grid]
            def call(*args,**kwargs):
                returned=launch(*args,**kwargs)
                observed.append(tuple(int(returned[1][name]) for name in ["BLOCK_M","BLOCK_N","BLOCK_K"]))
                return returned
            return call
    monkeypatch.setattr(module,"mm_kernel_general",Kernel())
    previous=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=tf32
    try:
        for m,n,resource_ok in [(64*sms,128,True),(32*sms,256,True),((sms+1)*64,128,False)]:
            a=torch.randn(m,128,device=flag_gems.device)
            b=torch.randn(128,n,device=flag_gems.device)
            values=[]
            for enabled in [False,True,False,True]:
                monkeypatch.setattr(module,"_GENERAL_WIDE_ENABLED",enabled)
                c=torch.empty(m,n,device=flag_gems.device)
                with torch.no_grad():module.general_mm(a,b,c,m,n,128)
                assert observed[-1]==((64,128,32) if enabled and not tf32 and resource_ok else (32,32,32))
                values.append(c)
            for value in values[1:]:assert torch.equal(value.view(torch.uint8),values[0].view(torch.uint8))
    finally:torch.backends.cuda.matmul.allow_tf32=previous
