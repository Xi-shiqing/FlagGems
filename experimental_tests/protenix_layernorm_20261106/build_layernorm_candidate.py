"""Correct normalization axes; split long parameter reductions under a resource cap."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'layernorm_baseline.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='00c6190b1074d151a65472486ab79411765eabb3186871e4d71fc1d4a799e4eb'
old='    M = input.shape[0]\n    N = input.numel() // M'
assert source.count(old)==1
fixed=source.replace(old,'''    N = math.prod(normalized_shape)
    M = input.numel() // N if N else 0
    if M == 0 or N == 0:
        return (torch.empty_like(input) if output_mask[0] else None,
                torch.zeros_like(weight) if output_mask[1] else None,
                torch.zeros_like(bias) if output_mask[2] else None)''')
(HERE/'layernorm_axes_fixed.py').write_text(fixed)
candidate=fixed.replace('from flag_gems.utils import libentry','from flag_gems.utils.device_info import get_sm_count\nfrom flag_gems.utils import libentry')
anchor='    with torch_device_fn.device(input.device):\n        weight_bias_backward_kernel[grid]('
assert candidate.count(anchor)==1
candidate=candidate.replace(anchor,'''    with torch_device_fn.device(input.device):
        if _split_parameter_gradients(grad_out, input, mean, rstd, weight_grad, bias_grad, M, N):
            return in_grad, weight_grad, bias_grad
        weight_bias_backward_kernel[grid](''')
candidate+='''

@libentry()
@triton.jit
def _parameter_partial_kernel(DY, X, MEAN, RSTD, PW, PB, M, N,
                              PARTS: tl.constexpr, HAS_W: tl.constexpr, HAS_B: tl.constexpr,
                              ROWS: tl.constexpr, COLS: tl.constexpr):
    col = ext.program_id(0)*COLS + tl.arange(0,COLS)
    part = ext.program_id(1)
    length = tl.cdiv(M,PARTS)
    start = part*length
    limit = tl.minimum(start+length,M)
    aw = tl.zeros((ROWS,COLS),tl.float32)
    ab = tl.zeros((ROWS,COLS),tl.float32)
    for offset in range(0,length,ROWS):
        row = start+offset+tl.arange(0,ROWS)
        valid = (row[:,None]<limit)&(col[None,:]<N)
        dy = tl.load(DY+row[:,None]*N+col[None,:],valid,other=0).to(tl.float32)
        if HAS_W:
            x = tl.load(X+row[:,None]*N+col[None,:],valid,other=0).to(tl.float32)
            mean = tl.load(MEAN+row,row<limit,other=0).to(tl.float32)
            rstd = tl.load(RSTD+row,row<limit,other=0).to(tl.float32)
            aw += dy*(x-mean[:,None])*rstd[:,None]
        if HAS_B:
            ab += dy
    if HAS_W:
        tl.store(PW+part*N+col,tl.sum(aw,axis=0),col<N)
    if HAS_B:
        tl.store(PB+part*N+col,tl.sum(ab,axis=0),col<N)


@libentry()
@triton.jit
def _parameter_merge_kernel(PW,PB,DW,DB,N,PARTS:tl.constexpr,
                            HAS_W:tl.constexpr,HAS_B:tl.constexpr,
                            REDUCE:tl.constexpr,COLS:tl.constexpr):
    col = ext.program_id(0)*COLS+tl.arange(0,COLS)
    row = tl.arange(0,REDUCE)
    valid = (row[:,None]<PARTS)&(col[None,:]<N)
    if HAS_W:
        values=tl.load(PW+row[:,None]*N+col[None,:],valid,other=0)
        tl.store(DW+col,tl.sum(values,axis=0),col<N)
    if HAS_B:
        values=tl.load(PB+row[:,None]*N+col[None,:],valid,other=0)
        tl.store(DB+col,tl.sum(values,axis=0),col<N)


def _split_parameter_gradients(dy,x,mean,rstd,dw,db,M,N):
    if M<4096 or x.dtype is not torch.float32 or x.device.type!='cuda':
        return False
    if any(t is not None and t.dtype is not torch.float32 for t in (dw,db)):
        return False
    columns=32
    desired=max(triton.cdiv(2*get_sm_count(),triton.cdiv(N,columns)),triton.cdiv(M,4096))
    maximum=min(256,M//128,(64*1024*1024)//(4*N*(int(dw is not None)+int(db is not None))))
    parts=min(desired,maximum)
    if parts<2:
        return False
    pw=torch.empty((parts,N),dtype=torch.float32,device=x.device) if dw is not None else None
    pb=torch.empty((parts,N),dtype=torch.float32,device=x.device) if db is not None else None
    _parameter_partial_kernel[(triton.cdiv(N,columns),parts)](
        dy,x,mean,rstd,pw,pb,M,N,parts,dw is not None,db is not None,128,columns,
        num_warps=4,num_stages=1)
    _parameter_merge_kernel[(triton.cdiv(N,columns),)](
        pw,pb,dw,db,N,parts,dw is not None,db is not None,triton.next_power_of_2(parts),columns,
        num_warps=4,num_stages=1)
    return True
'''
original_functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
new_functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(candidate).body if isinstance(n,ast.FunctionDef)}
changed=[n for n in original_functions if original_functions[n]!=new_functions[n]]
assert changed==['layer_norm_backward']
(HERE/'layernorm_candidate.py').write_text(candidate)
(HERE/'layernorm_candidate.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile='a/src/flag_gems/ops/layernorm.py',tofile='b/src/flag_gems/ops/layernorm.py')))
result={'baseline_sha256':hashlib.sha256(source.encode()).hexdigest(),'axes_fixed_sha256':hashlib.sha256(fixed.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'existing_changed_functions':changed,'new_functions':sorted(set(new_functions)-set(original_functions)),'original_kernels_unchanged':True,'production_default_changed':False}
(HERE/'layernorm_candidate_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
