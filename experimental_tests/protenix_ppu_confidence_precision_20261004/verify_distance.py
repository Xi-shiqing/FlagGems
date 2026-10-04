"""Numerical/layout/empty/broadcast controls and native autograd fallback."""
import json
from pathlib import Path

import torch
from euclidean_distance import euclidean_distance, NATIVE


def main():
    torch.manual_seed(101)
    records = []
    for dims in (1, 3, 5, 32, 129):
        for batch in (False, True):
            x = torch.randn((2, 17, dims*2) if batch else (17, dims*2),device='cuda')[..., ::2] + 4096
            y = torch.randn((1, 23, dims*2) if batch else (23, dims*2),device='cuda')[..., ::2] + 4096
            # Restore an explicitly noncontiguous view after translation.
            x, y = x.transpose(-2,-1).contiguous().transpose(-2,-1), y.transpose(-2,-1).contiguous().transpose(-2,-1)
            actual = euclidean_distance(x,y)
            gold = NATIVE(x.cpu().double(), y.cpu().double(), compute_mode='donot_use_mm_for_euclid_dist')
            error = (actual.cpu().double()-gold).abs()
            assert torch.allclose(actual.cpu().double(), gold,atol=2e-6,rtol=2e-6), (dims,batch,float(error.max()))
            same = euclidean_distance(x,x)
            assert torch.equal(same.diagonal(dim1=-2,dim2=-1),torch.zeros_like(same.diagonal(dim1=-2,dim2=-1)))
            records.append({'dims':dims,'broadcast':batch,'max_abs_error':float(error.max()),'status':'PASS'})
    for n,m,d in ((0,3,3),(3,0,3),(3,4,0)):
        x,y=torch.empty(n,d,device='cuda'),torch.empty(m,d,device='cuda')
        assert torch.equal(euclidean_distance(x,y),NATIVE(x,y,compute_mode='donot_use_mm_for_euclid_dist'))
    x=torch.randn(5,3,device='cuda',requires_grad=True)
    y=torch.randn(7,3,device='cuda',requires_grad=True)
    value=euclidean_distance(x,y)
    grads=torch.autograd.grad(value.sum(),(x,y))
    ref=NATIVE(x,y,compute_mode='donot_use_mm_for_euclid_dist')
    gold_grads=torch.autograd.grad(ref.sum(),(x,y))
    assert torch.equal(value,ref) and all(torch.equal(a,b) for a,b in zip(grads,gold_grads))
    result={'status':'PASS','cases':records,'empty_and_zero_dimension_controls':3,'native_autograd_fallback':'EXACT_PASS',
        'scope':'Forward kernel only; gradient-enabled operands use native direct distance, not a new Triton backward.'}
    Path(__file__).with_name('distance_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GENERAL_EUCLIDEAN_DISTANCE_CONTROLS_PASS',len(records))


if __name__=='__main__':main()
