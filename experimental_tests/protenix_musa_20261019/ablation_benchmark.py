"""Ablate bias fusion, softmax fusion, and in-place output separately."""
import json,os,random,statistics,time
from pathlib import Path
import torch,torch_musa,triton
import triton.language as tl
from amp_compat import install
install()
from protenix.model.triangular import layers
from bias_softmax import strides

@triton.jit
def kernel(S,O,B0,B1,H:tl.constexpr,Q:tl.constexpr,K:tl.constexpr,
           A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,A3:tl.constexpr,
           C0:tl.constexpr,C1:tl.constexpr,C2:tl.constexpr,C3:tl.constexpr,
           ADD:tl.constexpr,SOFTMAX:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);b=row//(H*Q);h=(row//Q)%H;q=row%Q;k=tl.arange(0,BLOCK)
    s=tl.load(S+row*K+k,k<K,0).to(tl.float32)
    if ADD:
        s=(s+tl.load(B0+b*A0+h*A1+q*A2+k*A3,k<K,0).to(tl.float32)).to(S.dtype.element_ty).to(tl.float32)
        s=(s+tl.load(B1+b*C0+h*C1+q*C2+k*C3,k<K,0).to(tl.float32)).to(S.dtype.element_ty).to(tl.float32)
    if SOFTMAX:
        s=tl.where(k<K,s,-float('inf'));e=tl.exp(s-tl.max(s,0));s=e/tl.sum(e,0)
    tl.store(O+row*K+k,s,k<K)

def attention(q,k,v,bias,variant):
    if variant=='native':return layers._attention(q,k,v,bias)
    score=torch.matmul(q,k.transpose(-1,-2));b,h,n,_=score.shape
    if variant=='softmax_only':
        for x in bias:score+=x
    out=torch.empty_like(score) if variant=='out_of_place' else score
    kernel[(b*h*n,)](score,out,*bias,h,n,n,*strides(bias[0],score.shape),*strides(bias[1],score.shape),
        ADD=variant!='softmax_only',SOFTMAX=variant!='bias_only',BLOCK=triton.next_power_of_2(n),num_warps=4,enable_fp_fusion=False)
    if variant=='bias_only':out=layers.softmax_no_cast(out,-1)
    return torch.matmul(out,v)

def run():
    root=Path(__file__).resolve().parent;torch.backends.mudnn.allow_tf32=True
    torch.manual_seed(20261018);torch.musa.manual_seed_all(20261018)
    assert torch.musa.mem_get_info()[1]-torch.musa.mem_get_info()[0]<1024**3
    variants=['native','bias_only','softmax_only','out_of_place','full_in_place'];rows=[]
    with torch.inference_mode():
        for n in (110,234,418,638,840,1012,1336,2053):
            batch=min(n,64)
            for dtype in (torch.bfloat16,torch.float32):
                for layout in ('contiguous','model_transposed'):
                    q,k,v=[torch.randn(batch,n,4,32,device='musa',dtype=dtype).transpose(1,2) for _ in range(3)]
                    q=q/(32**.5)
                    if layout=='contiguous':q=q.contiguous();k=k.contiguous();v=v.contiguous()
                    bias=[torch.zeros(batch,1,1,n,device='musa',dtype=torch.float32),torch.randn(1,4,n,n,device='musa',dtype=dtype)]
                    bias[0][...,-max(1,n//20):]=-1e4
                    reference=attention(q,k,v,bias,'native')
                    timings={x:[] for x in variants};mem={};errors={}
                    for x in variants:
                        y=attention(q,k,v,bias,x);torch.musa.synchronize()
                        errors[x]=dict(finite=bool(torch.isfinite(y).all().item()),relative_rms=float(torch.sqrt(torch.mean((y.float()-reference.float())**2)/torch.mean(reference.float()**2)).item()))
                        del y
                    for repeat in range(10):
                        order=variants.copy();random.Random(n+repeat).shuffle(order)
                        for x in order:
                            torch.musa.synchronize();torch.musa.reset_peak_memory_stats();start=time.perf_counter()
                            y=attention(q,k,v,bias,x);torch.musa.synchronize();elapsed=1000*(time.perf_counter()-start)
                            if repeat>=2:timings[x].append(elapsed)
                            mem[x]=torch.musa.max_memory_allocated();del y
                    for x in variants:
                        rows.append(dict(tokens=n,batch=batch,dtype=str(dtype),layout=layout,variant=x,
                            median_ms=statistics.median(timings[x]),samples_ms=timings[x],peak_allocated_bytes=mem[x],
                            **errors[x],numerical_pass=errors[x]['finite'] and errors[x]['relative_rms']<(.005 if dtype==torch.bfloat16 else .001)))
                    (root/'ablation.json').write_text(json.dumps(dict(source='Seeded synthetic tensors, model-like transpose layout, masked keys; not protein predictions',records=rows),indent=2))
                    del q,k,v,bias,reference
                    print('ABLATION',n,str(dtype),layout,flush=True)
    (root/'ablation_complete.json').write_text(json.dumps(dict(timestamp=time.time(),cases=len(rows),all_numerical_pass=all(x['numerical_pass'] for x in rows)),indent=2))

if __name__=='__main__':run()
