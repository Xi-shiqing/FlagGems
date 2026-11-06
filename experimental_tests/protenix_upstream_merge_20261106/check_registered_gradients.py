"""Check complete default registration and native autograd against CPU tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
sys.meta_path[:] = [x for x in sys.meta_path if '_flag_gems_editable' not in getattr(x, '__module__', '')]
import torch
import torch.nn.functional as F
import flag_gems

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
generator = torch.Generator().manual_seed(20261106)
result = {'package':flag_gems.__file__, 'checks':[], 'seed':20261106}

def check(name, operation, shapes):
    cpu = [torch.randn(s,generator=generator).requires_grad_() for s in shapes]
    gpu = [x.detach().to('cuda').requires_grad_() for x in cpu]
    ref = operation(*cpu)
    dy = torch.randn(ref.shape,generator=generator)
    ref.backward(dy)
    row = {'name':name}
    try:
        with flag_gems.use_gems(exclude=[]):
            actual = operation(*gpu)
            actual.backward(dy.to('cuda'))
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.detach().cpu(),ref.detach(),rtol=2e-5,atol=2e-5)
        differences = []
        for a,b in zip(gpu,cpu):
            assert a.grad is not None, 'missing input/parameter gradient'
            torch.testing.assert_close(a.grad.cpu(),b.grad,rtol=2e-5,atol=2e-5)
            diff=(a.grad.cpu().double()-b.grad.double()).norm().item()
            differences.append({'difference_l2':diff,'reference_l2':b.grad.double().norm().item()})
        row.update(status='PASS',gradients=differences)
    except Exception as exc:
        row.update(status='FAIL',error=str(exc)[:2500])
    result['checks'].append(row)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('REGISTERED_GRADIENT',name,row['status'],flush=True)

check('linear',lambda x,w,b:F.linear(x,w,b),[(3,17),(11,17),(11,)])
check('square',lambda x:(x*2).square(),[(3,7)])
check('mean',lambda x:x.mean(1),[(3,7,5)])
check('pad',lambda x:F.pad(x,(2,1),value=.5),[(3,7)])
check('sigmoid',torch.sigmoid,[(3,7)])
check('silu',F.silu,[(3,7)])
check('log_softmax',lambda x:torch.log_softmax(x,1),[(3,7)])
check('true_divide_inplace',lambda x:(x*2).true_divide_(4),[(3,7)])
check('views',lambda x:x.reshape(3,7,5).transpose(0,1).narrow(0,1,4),[(3,7,5)])
result['status']='PASS_REGISTERED_GRADIENTS' if all(x['status']=='PASS' for x in result['checks']) else 'FAIL_REGISTERED_GRADIENTS'
Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],flush=True)
