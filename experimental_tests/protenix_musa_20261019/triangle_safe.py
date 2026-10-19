"""Experimental fallback to the official out-of-place triangular update."""
import functools,os
from collections import Counter
import torch
from protenix.model.triangular.triangular import TriangleMultiplicativeUpdate
original=TriangleMultiplicativeUpdate.forward
hits=Counter()

def safe_update(module,z,mask,add,chunk,implementation):
    result=original(module,z,mask,inplace_safe=False,_add_with_inplace=False,
                    _inplace_chunk_size=chunk,triangle_multiplicative=implementation)
    return z+result if add else result

@functools.wraps(original)
def forward(self,z,mask=None,inplace_safe=False,_add_with_inplace=False,_inplace_chunk_size=256,triangle_multiplicative='torch'):
    if (not torch.is_grad_enabled() and z.device.type=='musa' and inplace_safe
            and z.shape[-2]>1024 and triangle_multiplicative=='torch'):
        hits['out_of_place_large']+=1
        return safe_update(self,z,mask,_add_with_inplace,_inplace_chunk_size,triangle_multiplicative)
    return original(self,z,mask,inplace_safe,_add_with_inplace,_inplace_chunk_size,triangle_multiplicative)

def install():TriangleMultiplicativeUpdate.forward=forward
