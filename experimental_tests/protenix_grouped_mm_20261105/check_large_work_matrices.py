"""All six saved long-K MM calls with larger outputs, including both zero cases."""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=Path('/mnt/cpfs/xishiqing/ppu_full_acceptance_20261005/precision_residual_20261019')
parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=2);args=parser.parse_args()
sys.path.insert(0,str(OLD.parent))
from bmm_card_gate_20261015ag import acquire
lock,reason,card=acquire('large_grouped_MM_20261105',HERE/'admission.json','exclusive',preferred=args.gpu)
try:
    os.environ['CUDA_VISIBLE_DEVICES']=str(card['gpu'])
    import torch,triton
    torch.set_num_threads(4);assert torch.__version__=='2.9.0' and triton.__version__=='3.6.0'
    sys.path.insert(0,str(OLD));from operator_replay_helpers_20261031AB import metrics
    modules={};identity={}
    paths={'original':OLD/'gems_canonical_softmax_v2/flag_gems/ops/mm.py','prior_large':OLD/'gems_longk_bounded_split_20261025/flag_gems/ops/mm.py','bounded_tree':OLD/'gems_bounded_reduction_v1/flag_gems/ops/mm.py','grouped64':Path('/root/xishiqing_protenix_precision_20261023/mm_accumulation_20261105_v8/grouped64_mm.py'),'grouped_work':Path('/root/xishiqing_protenix_precision_20261023/mm_accumulation_20261105_v9/grouped_work_mm.py')}
    for name,path in paths.items():
        spec=importlib.util.spec_from_file_location('flag_gems.ops.large_'+name,path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);modules[name]=module
        identity[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert identity['grouped64']['sha256']=='80708fa04ebb82becd7f2dd8879b626a1d8f0cd54c6871daf493de74da3efe06'
    state={'status':'RUNNING','card':card,'runtime_sources':identity,'cases':[],'new_H100_execution':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def digest(path):
        result=hashlib.sha256()
        with path.open('rb') as stream:
            for b in iter(lambda:stream.read(8*1024*1024),b''):result.update(b)
        return result.hexdigest()
    def identical(a,b):return a.detach().cpu().numpy().tobytes(order='C')==b.detach().cpu().numpy().tobytes(order='C')
    def quantize(v):
        bits=v.contiguous().view(torch.int32).to(torch.int64)
        return ((bits+4095+((bits>>13)&1))&0xFFFFE000).to(torch.int32).view(torch.float32)
    for entry in json.loads((HERE/'large_input_plan.json').read_text())['cases']:
        path=Path(entry['file']['path']);assert path.stat().st_size==entry['file']['bytes'] and digest(path)==entry['file']['sha256']
        payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True);t=payload['tensors'];meta=payload['metadata']
        ah,bh=t['A'],t['B'];print('LARGE_CPU_REFERENCE',entry['identifier'],list(ah.shape),list(bh.shape),flush=True)
        reference=ah.double()@bh.double();quantized=quantize(ah).double()@quantize(bh).double()
        a=torch.empty_strided(tuple(ah.shape),tuple(meta['A_stride']),device='cuda',dtype=torch.float32);a.copy_(ah)
        b=torch.empty_strided(tuple(bh.shape),tuple(meta['B_stride']),device='cuda',dtype=torch.float32);b.copy_(bh)
        for precision in ('fp32','tf32'):
            torch.backends.cuda.matmul.allow_tf32=precision=='tf32';row={'id':entry['identifier'],'family':entry['family'],'step':entry['step'],'precision':precision,'shapes':[list(ah.shape),list(bh.shape)],'input_sha256':entry['file']['sha256'],'implementations':{}}
            for name,module in modules.items():
                value=module.mm(a,b).cpu();assert torch.isfinite(value).all() and identical(value,module.mm(a,b).cpu())
                if name=='original' and meta['contextual_tf32_enabled']==(precision=='tf32'):assert identical(value,t['C']),'Original actual output was not recovered'
                row['implementations'][name]={'vs_CPU64':metrics(value,reference),'vs_RNE_operand_CPU64':metrics(value,quantized),'device_ms':[]}
                if name in ('grouped64','grouped_work'):row['implementations'][name]['partials']=module._long_reduction_splitk_parts(a,b,None,a.shape[0],b.shape[1],a.shape[1])
                for _ in range(2):module.mm(a,b)
            labels=list(modules)
            for iteration in range(5):
                order=labels[iteration%len(labels):]+labels[:iteration%len(labels)]
                for name in order:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();out=modules[name].mm(a,b);end.record();end.synchronize();row['implementations'][name]['device_ms'].append(start.elapsed_time(end))
            for value in row['implementations'].values():value['median_device_ms']=sorted(value['device_ms'])[2]
            assert identical(a.cpu(),ah) and identical(b.cpu(),bh)
            state['cases'].append(row);(HERE/'result.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');print('COMPLETE_LARGE_MM',entry['identifier'],precision,flush=True)
        del a,b,ah,bh,payload,t,reference,quantized,value,out
    assert len(state['cases'])==12;state['status']='COMPLETE_LARGE_REAL_MM_VALIDATION';(HERE/'result.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
finally:
    if lock.exists() and lock.read_text()==reason:lock.unlink()
