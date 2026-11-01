"""Scalar-only report for the failed subnormal equality gate; raw stays on PPU."""
from pathlib import Path
here=Path(__file__).resolve().parent
source=(here/'check_benchmark_fused_softmax_backward_20261024.py').read_text()
exec(compile(source[:source.index('\nresult={')],__file__,'exec'),globals())
import triton
import triton.language as tl
from triton.language.extra import libdevice
@triton.jit
def products(A,B,C,N:tl.constexpr,RN:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.arange(0,BLOCK);a=tl.load(A+i,i<N,other=0);b=tl.load(B+i,i<N,other=0)
 if RN:q=libdevice.mul_rn(a,b)
 else:q=a*b
 tl.store(C+i,q,i<N)
torch.set_flush_denormal(False)
def summary(a,b):
 a,b=a.detach().cpu().contiguous(),b.detach().cpu().contiguous();ai,bi=a.view(torch.int32),b.view(torch.int32)
 unequal=ai!=bi
 only_zero=(ai.to(torch.int64)&0x7fffffff)==0
 only_zero&=(bi.to(torch.int64)&0x7fffffff)==0
 return {'literal_bits_equal':torch.equal(ai,bi),'numeric_equal_CPU_flush_disabled':torch.equal(a,b),
  'different_bits':int(unequal.sum()),'mismatches_only_zero_sign':bool((~unequal|only_zero).all()),
  'actual_negative_zeros':int((ai==-2147483648).sum()),'reference_negative_zeros':int((bi==-2147483648).sum()),
  'max_abs_difference_float64':float((a.double()-b.double()).abs().max()),
  'actual_nonzero':int(torch.count_nonzero(a)),'reference_nonzero':int(torch.count_nonzero(b))}
report={'status':'RUNNING_TINY_GRADIENT_DIAGNOSIS','cases':[],
 'CPU_flush_disabled_for_scalar_analysis_only':True,'new_model_execution':False,'new_H100_execution':False}
raw={}
with flag_gems.use_gems(once=True),torch.no_grad():
 x=torch.arange(51,dtype=torch.float32,device='cuda').reshape(3,17)/11;p=module.softmax(x,-1)
 for index,value in enumerate((torch.finfo(torch.float32).tiny/4,torch.finfo(torch.float32).tiny,-torch.finfo(torch.float32).tiny/4,.125)):
  g=torch.full_like(p,value);q=torch.mul(g,p);plain=torch.empty_like(p);rn=torch.empty_like(p)
  products[(1,)](g,p,plain,g.numel(),False,triton.next_power_of_2(g.numel()))
  products[(1,)](g,p,rn,g.numel(),True,triton.next_power_of_2(g.numel()))
  a=module.softmax_backward(g,p,-1,torch.float32);b=reference.softmax_backward(g,p,-1,torch.float32)
  report['cases'].append({'upstream_value':value,'plain_product_vs_stored_Gems':summary(plain,q),
    'RN_product_vs_stored_Gems':summary(rn,q),'fused_backward_vs_unfused':summary(a,b)})
  for name,tensor in [('stored_product',q),('plain_product',plain),('RN_product',rn),('fused_gradient',a),('unfused_gradient',b)]:raw[str(index)+'/'+name]=tensor.cpu()
report['status']='COMPLETE_TINY_GRADIENT_DIAGNOSIS'
torch.save(raw,args.output_root/'tiny_gradient_raw_vectors.pt')
(args.output_root/'tiny_gradient_scalar_diagnosis.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(report['status'],flush=True)
