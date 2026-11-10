"""Time the whole allocating interface, using fixed ABBA blocks."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
import flag_gems
from flag_gems.utils import full_write_allocation, native_output_allocation

import inspect

candidate = flag_gems
assert flag_gems.vendor_name == 'thead', 'PPU FP32 fused activation benchmark'
full_write_allocation.ENABLED = native_output_allocation.ENABLED = True
torch.use_deterministic_algorithms(True)
torch.set_num_threads(4)
parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,required=True);args=parser.parse_args()
args.output_root.mkdir(parents=True,exist_ok=True)
p=args.output_root/'benchmark.json';assert not p.exists()
result={'status':'RUNNING','rows':[],'new_H100_execution':False,'graph_enabled':False,
        'scope':'Complete allocating interface under no_grad, including Python/dispatcher work. Separate per-call synchronization and 32-call batches; each path has 10 synchronous samples and 6 batches in ABBA blocks.',
        'fixtures':'Synthetic seeded inputs on several sizes, with broadcast and transpose layouts.'}
def save():p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def old(kind,values):
    a=F.silu(values[0]) if kind=='silu_mul' else torch.sigmoid(values[0])
    a=a*values[1]
    return a+values[2] if kind=='sigmoid_mul_add' else a
cases=[('small',(513,),(513,),False),('medium',(600,768),(600,768),False),
       ('diffusion',(5,600,768),(5,600,768),False),('large',(600,600,128),(600,600,128),False),
       ('broadcast',(1,600,128),(5,600,128),False),('transpose',(128,256),(128,256),True)]
save()
try:
    with torch.no_grad(),flag_gems.use_gems():
        for index,(name,a_shape,b_shape,transposed) in enumerate(cases):
            generator=torch.Generator(device="cpu").manual_seed(101+index)
            values=[torch.randn(shape,generator=generator,device='cpu',dtype=torch.float32).to(flag_gems.device) for shape in [a_shape,b_shape,b_shape]]
            if transposed:values=[x.transpose(-1,-2) for x in values]
            input_hashes=[sha(x) for x in values]
            for precision in ['FP32','TF32']:
                torch.backends.cuda.matmul.allow_tf32=precision=='TF32'
                torch.backends.cudnn.allow_tf32=precision=='TF32'
                for kind in ['sigmoid_mul','sigmoid_mul_add','silu_mul']:
                    inputs=values if kind=='sigmoid_mul_add' else values[:2]
                    functions={'control':lambda:old(kind,inputs),'candidate':lambda:getattr(candidate,kind)(*inputs)}
                    outputs={label:sha(fn()) for label,fn in functions.items()}
                    assert len(set(outputs.values()))==1,(name,precision,kind)
                    samples={label:[] for label in functions};batches={label:[] for label in functions}
                    for label in ['control','candidate','candidate','control']:
                        fn=functions[label]
                        for _ in range(3):value=fn()
                        torch.cuda.synchronize()
                        for _ in range(5):
                            start=time.perf_counter_ns();value=fn();torch.cuda.synchronize()
                            samples[label].append((time.perf_counter_ns()-start)/1000)
                        for _ in range(3):
                            torch.cuda.synchronize();start=time.perf_counter_ns()
                            for _ in range(32):value=fn()
                            torch.cuda.synchronize();batches[label].append((time.perf_counter_ns()-start)/32000)
                    assert input_hashes==[sha(x) for x in values]
                    result['rows'].append({'case':name,'precision':precision,'kind':kind,
                       'input_shapes':[list(x.shape) for x in inputs],'input_strides':[list(x.stride()) for x in inputs],
                       'input_sha256':input_hashes[:len(inputs)],'output_bytes_equal':True,
                       'synchronized_call_samples_us':samples,'batch_mean_samples_us':batches,
                       'synchronized_call_median_us':{k:statistics.median(v) for k,v in samples.items()},
                       'batch_mean_median_us':{k:statistics.median(v) for k,v in batches.items()}});save()
    assert len(result['rows'])==36
    result.update(status='PASS',case_count=len(result['rows']),source_sha256=hashlib.sha256(Path(inspect.getsourcefile(candidate.sigmoid_mul)).read_bytes()).hexdigest())
    save();print(json.dumps({'status':'PASS','case_count':len(result['rows'])}))
except BaseException as error:
    result.update(status='FAILED_PRESERVED',error_type=type(error).__name__);save();raise
