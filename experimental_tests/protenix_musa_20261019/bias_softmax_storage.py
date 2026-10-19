"""Storage ablation: separate output, or BF16 separate / FP32 in-place."""
from collections import Counter
import torch,triton
from protenix.model.triangular import layers
from bias_softmax import original,strides,fused_attention as inplace_attention
from ablation_benchmark import kernel
hits=Counter()

def separate(query,key,value,biases):
    if (torch.is_grad_enabled() or query.device.type!='musa' or query.ndim!=4
            or len(biases)!=2 or query.dtype not in (torch.float32,torch.bfloat16)):
        return original(query,key,value,biases)
    shape=tuple(query.shape[:-1])+(key.shape[-2],)
    steps=[strides(b,shape) for b in biases]
    if shape[-1]<200 or shape[-1]>4096 or any(s is None for s in steps):
        return original(query,key,value,biases)
    scores=torch.matmul(query,key.transpose(-1,-2))
    if not scores.is_contiguous():return original(query,key,value,biases)
    probability=torch.empty_like(scores);b,h,q,k=scores.shape
    kernel[(b*h*q,)](scores,probability,*biases,h,q,k,*steps[0],*steps[1],
        ADD=True,SOFTMAX=True,BLOCK=triton.next_power_of_2(k),num_warps=4,enable_fp_fusion=False)
    hits[f'separate_{str(scores.dtype)}_K{k}']+=1
    return torch.matmul(probability,value)

def hybrid(query,key,value,biases):
    if query.dtype==torch.bfloat16:
        hits['hybrid_bf16_separate']+=1
        return separate(query,key,value,biases)
    hits['hybrid_fp32_inplace']+=1
    return inplace_attention(query,key,value,biases)
