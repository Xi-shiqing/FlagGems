"""Actual registered backward, literal-bit equality and complete API timings."""
import argparse,ast,hashlib,importlib,importlib.util,json,sys,time,statistics
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,required=True);args=parser.parse_args()
args.output_root.mkdir(parents=True,exist_ok=False);here=Path(__file__).resolve().parent
root=here/'gems_fused_softmax_backward_20261024';manifest=json.loads((root/'fused_candidate_manifest.json').read_text())
assert 'flag_gems' not in sys.modules
for finder in list(sys.meta_path):
 if type(finder).__module__=='_flag_gems_editable':sys.meta_path.remove(finder)
sys.path.insert(0,str(root))
import torch,flag_gems
module=importlib.import_module('flag_gems.ops.softmax');torch.set_num_threads(2)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert Path(module.__file__).resolve()==(root/'flag_gems/ops/softmax.py').resolve() and sha(Path(module.__file__))==manifest['candidate_softmax_sha256']
spec=importlib.util.spec_from_file_location('_fixed_unfused_backward_reference',here/'gems_noninner_backward_canonical_20261024/flag_gems/ops/softmax.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
assert sha(Path(reference.__file__))==manifest['fixed_unfused_softmax_sha256']
tree=ast.parse((here/'probe_softmax_shift_stability_20261023.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='metric'];assert len(nodes)==1
exec(compile(ast.Module(body=nodes,type_ignores=[]),__file__,'exec'),globals())

def bits(a,b):
 assert a.dtype==b.dtype and a.shape==b.shape
 dtype=torch.int32 if a.dtype==torch.float32 else torch.int16
 return torch.equal(a.detach().contiguous().view(dtype),b.detach().contiguous().view(dtype))

result={'status':'RUNNING_FUSED_SOFTMAX_BACKWARD_CHECKS','manifest':manifest,'new_model_execution':False,'new_H100_execution':False,
 'shape_allowlist':False,'real':{},'synthetic':[],'benchmarks':[],
 'timing_scope':'Seven paired rounds of ten warmed complete public Gems backward API calls; include gradient contiguous copy, product materialization when present and result allocation. Device event/10 and host wall/10. Not isolated kernel or model/MFU timing. H100 comparable timing unavailable.'}
fixture=here.parent/'train_backward_profile_20261018/h100_sftp_references/more_component_fixtures/component_fixtures.pt'
assert sha(fixture)=='fa97504176660987762e5b55a0e59a7c252f07b9e48279e2cfb4b02a7f2301d7'
pack=torch.load(fixture,map_location='cpu',mmap=True,weights_only=False)
real=pack['fixtures']['model.msa_module.blocks.0.msa_stack.msa_pair_weighted_averaging.softmax_w']
# Preserve all completed numerical evidence before entering the timing stage.
def save(): (args.output_root/'fused_softmax_backward_checks_and_benchmarks.json').write_text(json.dumps(result,indent=2,allow_nan=False))
with flag_gems.use_gems(once=True):
 for precision in ('fp32','tf32'):
  torch.backends.cuda.matmul.allow_tf32=precision=='tf32'
  x=real['args'][0].cuda().requires_grad_(True);g=real['output_gradients'][''].cuda()
  p=torch.softmax(x,-2);assert bits(p,reference.softmax(x.detach(),-2))
  with torch.no_grad():expected=reference.softmax_backward(g,p,-2,torch.float32)
  a=torch.autograd.grad(p,x,g,retain_graph=True)[0];b=torch.autograd.grad(p,x,g)[0]
  assert bits(a,expected) and bits(a,b),'Real complete VJP differs from fixed unfused source'
  result['real'][precision]={'literal_bits_equal_to_unfused':True,'complete_VJP':metric(a,expected),'forward_literal_bits_equal':True}
 shapes=[((3,1,5),1),((2,17,3),1),((2,65,33),1),((2,129,129),1),((2,1025,3),1),((5,7,3),0),((2,33,1025),1),((12416,8),1),((17,129),1),((3,2049),1),((2,7,437),2)]
 for dtype,tolerance in ((torch.float32,2e-6),(torch.float16,2e-3),(torch.bfloat16,2e-2)):
  for shape,axis in shapes:
   n=1
   for d in shape:n*=d
   cpu=torch.sin(torch.arange(n,dtype=torch.float32).reshape(shape)/11).to(dtype)
   x=cpu.cuda().requires_grad_(True);storage=torch.cos(torch.arange(n*2,dtype=torch.float32).reshape(*shape[:-1],shape[-1]*2)/17).to(dtype).cuda();g=storage[...,::2]
   p=torch.softmax(x,axis);assert bits(p,reference.softmax(x.detach(),axis))
   with torch.no_grad():expected=reference.softmax_backward(g,p,axis,dtype)
   a=torch.autograd.grad(p,x,g,retain_graph=True)[0];b=torch.autograd.grad(p,x,g)[0]
   assert bits(a,expected) and bits(a,b),(dtype,shape,'Literal VJP bits differ')
   pd=cpu.double().softmax(axis);gd=g.cpu().double();oracle=pd*(gd-(pd*gd).sum(axis,keepdim=True))
   score=metric(a,oracle);assert score['relative_l2'] is not None and score['relative_l2']<tolerance
   result['synthetic'].append({'dtype':str(dtype),'shape':list(shape),'axis':axis,'literal_bits_equal_to_unfused':True,'vs_CPU64':score,'declared_tolerance_ratio':tolerance})
   save()
 # Explicit subnormal gradients must preserve the unfused multiplication behavior.
 for value in (torch.finfo(torch.float32).tiny/4,torch.finfo(torch.float32).tiny,-torch.finfo(torch.float32).tiny/4):
  x=torch.arange(51,dtype=torch.float32,device='cuda').reshape(3,17)/11;p=module.softmax(x,-1);g=torch.full_like(p,value)
  with torch.no_grad():a=module.softmax_backward(g,p,-1,torch.float32);b=reference.softmax_backward(g,p,-1,torch.float32)
  assert bits(a,b),'Subnormal backward product differs'
 result['subnormal_gradient_literal_bit_checks']=3
 result['status']='COMPLETE_FUSED_SOFTMAX_BACKWARD_NUMERICAL_CHECKS';save()
 # Isolate actual Python-level product materialization, without affecting results.
 old_mul=torch.mul;calls={'reference':0,'fused':0};active=[None]
 def counted_mul(*v,**kw):
  if active[0] is not None:calls[active[0]]+=1
  return old_mul(*v,**kw)
 torch.mul=counted_mul
 try:
  x=real['args'][0].cuda();p=module.softmax(x,-2);g=real['output_gradients'][''].cuda()
  with torch.no_grad():
   active[0]='reference';reference.softmax_backward(g,p,-2,torch.float32)
   active[0]='fused';module.softmax_backward(g,p,-2,torch.float32)
  assert calls=={'reference':1,'fused':0},calls
 finally:torch.mul=old_mul
 result['materialized_Torch_mul_calls_per_real_backward']=calls
 cases=[('real_MSA',real['args'][0],real['output_gradients'][''],-2),('synthetic_inner8',torch.sin(torch.arange(12416*8).reshape(12416,8)/11),None,-1),('synthetic_inner129',torch.sin(torch.arange(257*129).reshape(257,129)/11),None,-1),('synthetic_inner2049',torch.sin(torch.arange(3*2049).reshape(3,2049)/11),None,-1)]
 for name,cpu,cpu_g,axis in cases:
  x=cpu.float().cuda();p=module.softmax(x,axis);g=cpu_g.cuda() if cpu_g is not None else torch.cos(x*1.7)
  for precision in ('fp32','tf32'):
   torch.backends.cuda.matmul.allow_tf32=precision=='tf32';providers={'reference':reference.softmax_backward,'fused':module.softmax_backward};rounds={k:[] for k in providers}
   with torch.no_grad():
    for f in providers.values():
     for unused in range(4):f(g,p,axis,torch.float32)
    for ri in range(7):
     for kind in (list(providers) if ri%2==0 else list(reversed(providers))):
      f=providers[kind];torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
      t=time.perf_counter();start.record()
      for unused in range(10):out=f(g,p,axis,torch.float32)
      end.record();end.synchronize();rounds[kind].append({'API_device_ms':start.elapsed_time(end)/10,'API_host_wall_ms':(time.perf_counter()-t)*1000/10})
   result['benchmarks'].append({'name':name,'shape':list(x.shape),'axis':axis,'model_matrix_precision_flag':precision,'rounds':rounds,'medians':{k:{m:statistics.median(v[m] for v in rows) for m in rows[0]} for k,rows in rounds.items()}});save()
result['status']='COMPLETE_FUSED_SOFTMAX_BACKWARD_NUMERICAL_AND_API_BENCHMARKS';save();print(result['status'],len(result['synthetic']),flush=True)
