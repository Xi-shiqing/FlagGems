"""A prefilled output must be fully replaced, with grad and layout preserved."""
import importlib
import pytest
import torch
import flag_gems
from flag_gems.utils import native_output_allocation as allocator

pytestmark = pytest.mark.skipif(not allocator.ENABLED, reason="requires optional allocator")

@pytest.fixture
def allocations(monkeypatch):
    native=allocator._empty
    class Allocations(list):
        boolean_poison = True
    calls=Allocations()
    def poisoned(shape,dtype,device):
        output=native(shape,dtype,device)
        output.fill_(float("nan") if dtype.is_floating_point else calls.boolean_poison if dtype==torch.bool else 97)
        calls.append(output)
        return output
    monkeypatch.setattr(allocator,"_empty",poisoned)
    before=(torch.are_deterministic_algorithms_enabled(),torch.utils.deterministic.fill_uninitialized_memory)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    yield calls
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(before[0])
    torch.utils.deterministic.fill_uninitialized_memory=before[1]

@pytest.mark.parametrize("op",["zeros","ones","full","zeros_like","ones_like","full_like","new_zeros","new_full"])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float16,torch.bfloat16,torch.int32,torch.int64,torch.bool])
@pytest.mark.parametrize("grad",[False,True])
def test_factory_overwrite(allocations,op,dtype,grad):
    value=3 if "full" in op else 1 if "ones" in op else 0
    allocations.boolean_poison=not bool(value)
    device=flag_gems.device
    x=torch.empty((7,19),dtype=dtype,device=device).t()
    function=getattr(importlib.import_module("flag_gems.ops."+op),op)
    args=([7,19],) if op in ("zeros","ones","full") else (x,[7,19]) if op.startswith("new_") else (x,)
    args=args+(3,) if "full" in op else args
    kwargs={"dtype":dtype,"device":x.device}
    with torch.set_grad_enabled(grad):output=function(*args,**kwargs)
    value=3 if "full" in op else 1 if "ones" in op else 0
    assert torch.equal(output.cpu(),torch.full(output.shape,value,dtype=dtype))
    if op in ("ones_like","zeros_like"):assert output.stride()==x.stride()
    assert len(allocations)==(0 if grad else 1)
    if output.numel():assert output.data_ptr()!=x.data_ptr()

@pytest.mark.parametrize("shape",[(),(0,),(2,0,7)])
@pytest.mark.parametrize("op",["zeros","ones","full","new_zeros"])
def test_factory_zero_size(allocations,shape,op):
    function=getattr(importlib.import_module("flag_gems.ops."+op),op)
    x=torch.empty((),device=flag_gems.device)
    args=(x,shape) if op=="new_zeros" else (shape,)
    if op=="full":args=args+(2,)
    with torch.no_grad():output=function(*args,device=x.device,dtype=x.dtype)
    assert tuple(output.shape)==shape
    if not shape:assert output.item()==(2 if op=="full" else 1 if op=="ones" else 0)

@pytest.mark.parametrize("mode",["constant","reflect","replicate","circular"])
@pytest.mark.parametrize("layout",["contiguous","transpose","slice"])
def test_padding_overwrite(allocations,mode,layout):
    x=torch.arange(2*5*7,dtype=torch.float32).reshape(2,5,7)
    if layout=="transpose":x=x.transpose(1,2)
    if layout=="slice":x=x[...,::2]
    kwargs={"value":2.5} if mode=="constant" else {}
    expected=torch.nn.functional.pad(x,(1,2),mode=mode,**kwargs)
    with torch.no_grad():actual=flag_gems.pad(x.to(flag_gems.device),(1,2),mode=mode,**kwargs)
    assert torch.equal(actual.cpu(),expected)
    assert len(allocations)==1

@pytest.mark.parametrize("dim",[0,1,2])
def test_cat_overwrite(allocations,dim):
    values=[torch.arange(3*5*7).reshape(3,5,7).float().transpose(1,2)+i for i in range(6)]
    expected=torch.cat(values,dim)
    with torch.no_grad():actual=flag_gems.cat([v.to(flag_gems.device) for v in values],dim)
    assert torch.equal(actual.cpu(),expected)
    assert len(allocations)>=1

@pytest.mark.parametrize("layout",["contiguous","transpose","slice"])
def test_index_overwrite(allocations,layout):
    x=torch.arange(3*7*9).reshape(3,7,9).float()
    if layout=="transpose":x=x.transpose(0,1)
    if layout=="slice":x=x[...,::2]
    idx=torch.tensor([0,-1,1])
    expected=x[idx]
    with torch.no_grad():actual=flag_gems.index(x.to(flag_gems.device),[idx.to(flag_gems.device)])
    assert torch.equal(actual.cpu(),expected)
    assert len(allocations)==0

@pytest.mark.parametrize("kind",["all","all_dim","all_dims"])
@pytest.mark.parametrize("value",[0,1])
def test_all_overwrite(allocations,kind,value):
    x=torch.full((7,19,23),value,dtype=torch.float32)
    allocations.boolean_poison=not bool(value)
    expected=x.bool().all() if kind=="all" else x.bool().all(1) if kind=="all_dim" else x.bool().all((0,2))
    function=getattr(importlib.import_module("flag_gems.ops.all"),kind)
    args=() if kind=="all" else (1,) if kind=="all_dim" else ([0,2],)
    with torch.no_grad():actual=function(x.to(flag_gems.device),*args)
    assert torch.equal(actual.cpu(),expected)
    assert len(allocations)==(2 if kind=="all" else 1)

@pytest.mark.parametrize("bias",[False,True])
def test_conv_output_overwrite(allocations,bias):
    torch.manual_seed(20261110)
    x=torch.randn(1,16,7,9);w=torch.randn(8,16,1,1);b=torch.randn(8) if bias else None
    expected=torch.nn.functional.conv2d(x,w,b)
    with torch.no_grad():actual=flag_gems.conv2d(x.to(flag_gems.device),w.to(flag_gems.device),None if b is None else b.to(flag_gems.device))
    torch.testing.assert_close(actual.cpu(),expected,rtol=2e-5,atol=2e-5)
    assert len(allocations)>=1


def test_native_allocation_stays_off_with_grad(allocations):
    x=torch.arange(7*19,device=flag_gems.device).reshape(7,19).float()
    with torch.enable_grad():
        flag_gems.constant_pad_nd(x,(1,2),0)
        flag_gems.cat([x,x],0)
        flag_gems.index(x,[torch.tensor([0,1],device=x.device)])
        flag_gems.all(x)
    assert not allocations


@pytest.mark.parametrize("operation",["pad","conv2d"])
def test_custom_autograd_forward_and_backward(allocations,operation):
    torch.manual_seed(20261110)
    x=torch.randn(1,16,7,9,requires_grad=True)
    inputs=[x]
    if operation=="conv2d":
        inputs += [torch.randn(8,16,1,1,requires_grad=True),torch.randn(8,requires_grad=True)]
    gpu=[v.detach().to(flag_gems.device).requires_grad_() for v in inputs]
    with torch.enable_grad():
        if operation=="pad":
            expected=torch.nn.functional.pad(x,(1,2),value=2.5)
            actual=flag_gems.constant_pad_nd(gpu[0],(1,2),2.5)
        else:
            expected=torch.nn.functional.conv2d(*inputs)
            actual=flag_gems.conv2d(*gpu)
        expected.square().sum().backward()
        actual.square().sum().backward()
    torch.testing.assert_close(actual.cpu(),expected,rtol=2e-5,atol=2e-5)
    for a,b in zip(gpu,inputs):
        torch.testing.assert_close(a.grad.cpu(),b.grad,rtol=3e-5,atol=3e-5)
    assert allocations


@pytest.mark.parametrize("layout",["transpose","slice"])
@pytest.mark.parametrize("indices",["adjacent","separated","middle"])
def test_generic_index_negative_values(allocations,layout,indices):
    x=torch.arange(5*7*9).reshape(5,7,9).float()
    if layout=="transpose":x=x.transpose(0,1)
    else:x=x[...,::2]
    a,b=torch.tensor([0,-1,1]),torch.tensor([-1,0,1])
    ind=[a,b] if indices=="adjacent" else [a,None,b] if indices=="separated" else [None,a]
    expected=x[tuple(slice(None) if i is None else i for i in ind)]
    with torch.no_grad():
        actual=flag_gems.index(x.to(flag_gems.device),[None if i is None else i.to(flag_gems.device) for i in ind])
    assert torch.equal(actual.cpu(),expected)
