"""Check the actual ATen registration, including autograd and nondefault streams."""
import importlib
import pytest
import torch
import flag_gems
from flag_gems.utils import native_output_allocation as allocator

pytestmark=pytest.mark.skipif(not allocator.ENABLED,reason="requires optional native allocator")

@pytest.mark.parametrize("name",["zeros","ones","full","zeros_like","ones_like","full_like","new_zeros","new_full"])
@pytest.mark.parametrize("dtype",[torch.float32,torch.int64])
@pytest.mark.parametrize("grad",[False,True])
def test_registered_factory(monkeypatch,name,dtype,grad):
    x=torch.empty((7,19),device=flag_gems.device,dtype=dtype).t()
    args=(x.shape,) if name in ['zeros','ones','full'] else (x.shape,) if name.startswith('new_') else (x,)
    if 'full' in name:args=args+(3,)
    f=getattr(x,name) if name.startswith('new_') else getattr(torch,name)
    native=allocator._empty;calls=[]
    def poisoned(*values):
        output=native(*values);output.fill_(float('nan') if output.dtype.is_floating_point else 97);calls.append(output);return output
    monkeypatch.setattr(allocator,'_empty',poisoned)
    stream=torch.cuda.Stream()
    with torch.set_grad_enabled(grad),torch.cuda.stream(stream),flag_gems.use_gems():
        output=f(*args,device=x.device,dtype=dtype)
        factory_calls=len(calls)
        consumer=output+1
    stream.synchronize()
    value=3 if 'full' in name else 1 if 'ones' in name else 0
    assert torch.equal(output.cpu(),torch.full(output.shape,value,dtype=dtype))
    assert bool(factory_calls)==(not grad)
    assert torch.equal(consumer.cpu(),torch.full(output.shape,value+1,dtype=dtype))
