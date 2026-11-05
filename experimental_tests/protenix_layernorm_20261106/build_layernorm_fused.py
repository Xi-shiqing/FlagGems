"""Fuse dX and partial affine gradients to read each input tile once."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'layernorm_candidate_v2.py').read_text()
anchor='    if output_mask[0]:\n        in_grad = torch.empty_like(input)'
assert source.count(anchor)==1
source=source.replace(anchor,'''    fused = _fused_large_row_backward(grad_out, input, mean, rstd, weight, bias, output_mask, M, N)
    if fused is not None:
        return fused

'''+anchor)
source+='''

@libentry()
@triton.jit
def _fused_input_parameter_kernel(DY,X,MEAN,RSTD,W,DX,PW,PB,M,N,
                                  PARTS:tl.constexpr,HAS_W:tl.constexpr,HAS_B:tl.constexpr,
                                  ROWS:tl.constexpr,COLS:tl.constexpr):
    part=tle.program_id(0)
    col=tl.arange(0,COLS)
    length=tl.cdiv(M,PARTS)
    start=part*length
    limit=tl.minimum(start+length,M)
    weight=tl.load(W+col,col<N,other=0).to(tl.float32) if W is not None else 1.
    aw=tl.zeros((COLS,),tl.float32)
    ab=tl.zeros((COLS,),tl.float32)
    for offset in range(0,length,ROWS):
        row=start+offset+tl.arange(0,ROWS)
        valid=(row[:,None]<limit)&(col[None,:]<N)
        dy=tl.load(DY+row[:,None]*N+col[None,:],valid,other=0).to(tl.float32)
        x=tl.load(X+row[:,None]*N+col[None,:],valid,other=0).to(tl.float32)
        mean=tl.load(MEAN+row,row<limit,other=0).to(tl.float32)
        rstd=tl.load(RSTD+row,row<limit,other=0).to(tl.float32)
        centered=tl.where(valid,x-mean[:,None],0.)
        z=centered*rstd[:,None]
        g=dy*weight
        g=tl.where(valid,g,0.)
        a=tl.sum(g,axis=1)/N
        b=tl.sum(g*z,axis=1)/N
        dx=rstd[:,None]*(g-a[:,None]-z*b[:,None])
        tl.store(DX+row[:,None]*N+col[None,:],dx,valid)
        if HAS_W:
            aw+=tl.sum(dy*centered*rstd[:,None],axis=0)
        if HAS_B:
            ab+=tl.sum(dy,axis=0)
    if HAS_W:
        tl.store(PW+part*N+col,aw,col<N)
    if HAS_B:
        tl.store(PB+part*N+col,ab,col<N)


def _fused_large_row_backward(dy,x,mean,rstd,weight,bias,mask,M,N):
    if (not mask[0] or not (mask[1] or mask[2]) or M<4096 or N>2048
            or x.dtype is not torch.float32 or x.device.type!='cuda'):
        return None
    if any(t is not None and t.dtype is not torch.float32 for t in (weight,bias)):
        return None
    maximum=min(256,M//128,(64*1024*1024)//(4*N*(int(mask[1])+int(mask[2]))))
    parts=min(max(4*get_sm_count(),triton.cdiv(M,4096)),maximum)
    if parts<2:
        return None
    cols=triton.next_power_of_2(N)
    rows=min(32,max(1,2048//cols))
    dx=torch.empty_like(x)
    dw=torch.empty_like(weight) if mask[1] else None
    db=torch.empty_like(bias) if mask[2] else None
    pw=torch.empty((parts,N),dtype=torch.float32,device=x.device) if mask[1] else None
    pb=torch.empty((parts,N),dtype=torch.float32,device=x.device) if mask[2] else None
    with torch_device_fn.device(x.device):
        _fused_input_parameter_kernel[(parts,)](
            dy,x,mean,rstd,weight,dx,pw,pb,M,N,parts,mask[1],mask[2],rows,cols,
            num_warps=4,num_stages=1)
        _parameter_merge_kernel[(triton.cdiv(N,32),)](
            pw,pb,dw,db,N,parts,mask[1],mask[2],triton.next_power_of_2(parts),32,
            num_warps=4,num_stages=1)
    return dx,dw,db
'''
ast.parse(source)
(HERE/'layernorm_candidate_v3.py').write_text(source)
(HERE/'layernorm_fused_manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'reason':'Measured separate input-gradient kernels dominate the current split-only candidate; fuse input reads with partial affine reduction.','row_tile_elements_cap':2048,'partial_workspace_cap_bytes':64*1024**2,'maximum_parts':256,'maximum_normalized_elements_for_fusion':2048,'production_default_changed':False},indent=2)+'\n')
print(hashlib.sha256(source.encode()).hexdigest())
