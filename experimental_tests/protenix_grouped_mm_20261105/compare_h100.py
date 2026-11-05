"""CPU vector comparison using the eight explicitly authorized existing H100 dW files."""
import argparse,hashlib,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
parser=argparse.ArgumentParser();parser.add_argument('--result-root',type=Path,required=True);parser.add_argument('--h100-root',type=Path,default=Path(__file__).parent);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
import torch
torch.set_num_threads(4)
sys.path.insert(0,'/mnt/cpfs/xishiqing/ppu_full_acceptance_20261005/precision_residual_20261019')
from operator_replay_helpers_20261031AB import metrics

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
state=json.loads((args.result_root/'result.json').read_text());assert state['status']=='COMPLETE'
manifest=json.loads((args.h100_root/'h100_existing_manifest.json').read_text());assert len(manifest['files'])==8
rows=[]
for row in state['cases']:
    mode,source,step=row['execution_precision'],row['input_precision'],row['step']
    entry=next(v for v in manifest['files'] if v['execution_precision']==mode and v['source_precision']==source and v['step']==step)
    assert entry['source_packet_sha256']==row['input_sha256'] and entry['call']==row['call']
    hpath=args.h100_root/'h100_existing'/entry['file'];assert hpath.stat().st_size==entry['bytes'] and digest(hpath)==entry['sha256']
    h=torch.load(hpath,map_location='cpu',weights_only=False)
    for label,value in row['implementations'].items():
        path=Path(value['output_file']);assert digest(path)==value['output_sha256']
        p=torch.load(path,map_location='cpu',weights_only=False)
        rows.append(dict(input_precision=source,execution_precision=mode,step=step,call=row['call'],implementation=label,
                         PPU_vs_native_H100=metrics(p,h),PPU_vs_CPU64=value['vs_CPU64'],H100_vs_CPU64=entry['vs_CPU64'],
                         PPU_RNE_accumulation_error=value['vs_RNE_operand_CPU64'],median_API_device_ms=value['median_device_ms'],
                         partials=value['parts'],partial_workspace_bytes=(value['parts'] or 0)*p.numel()*4,
                         input_sha256=row['input_sha256'],PPU_output_sha256=digest(path),H100_output_sha256=entry['sha256']))
result={'status':'COMPLETE_CPU_VECTOR_COMPARISON','rows':rows,'CUDA_initialized':torch.cuda.is_initialized(),'raw_tensors_downloaded_to_local':False}
assert not result['CUDA_initialized']
args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':result['status'],'rows':len(rows),'output':str(args.output),'CUDA_initialized':False}))
