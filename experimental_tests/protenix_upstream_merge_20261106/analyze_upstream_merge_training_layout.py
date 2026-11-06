"""Compare step1 gradients and updates against cached H100; step0 Adam against old PPU."""
import argparse,ast,hashlib,json,math,os
from collections import defaultdict
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
torch.set_num_threads(4)
HERE=Path('/mnt/cpfs/xishiqing/ppu_full_acceptance_20261005/precision_residual_20261019')
L=Path('/root/xishiqing_protenix_precision_20261023')
parser=argparse.ArgumentParser();parser.add_argument('--precision',choices=('fp32','tf32'),required=True);parser.add_argument('--output-root',type=Path,required=True);args=parser.parse_args();args.output_root.mkdir(exist_ok=False)

def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
for filename,names in [('analyze_linear_forward_training_20261023.py',('rng','finish','metrics')),('analyze_softmax_forward_fidelity_training_20261023.py',('moments',))]:
    nodes=[node for node in ast.parse((HERE/filename).read_text()).body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(HERE/filename),'exec'),globals())

def compact(value):
    result={'total':value['total'],'modules':{k:v for k,v in value['modules'].items() if '.' not in k or k=='diffusion_module.diffusion_transformer'},'top10_parameters':value['top40_parameters_by_squared_difference'][:10]}
    for item in [result['total'],*result['modules'].values(),*result['top10_parameters']]:item['numeric_equal']=item.pop('bitwise_equal')
    return result
folder=L/('upstream_merge_training_'+args.precision+'_20261106_layout_fix')
base=L/('fused_backward_v2_'+args.precision+'_r0_20261024')
h=HERE/('h100_sftp_references/matching_native_'+args.precision+'_r0')
read=lambda p:json.loads(p.read_text())
reports=[read(d/'training_report.json') for d in (folder,base,h)]
assert all(r['status']=='pass' and len(r['losses'])==len(r['batch_audits'])==2 for r in reports)
assert reports[0]['batch_audits']==reports[1]['batch_audits']==reports[2]['batch_audits']
assets=lambda r:{Path(k).name:v for k,v in r['input_hashes'].items()}
assert assets(reports[0])==assets(reports[1])==assets(reports[2])
random=[rng(d) for d in (folder,base,h)]
assert random[0]==random[1]==random[2] and random[0]['count']==3338
audit=read(folder/'actual_parameter_audit.json');assert audit['audit_step']==1 and audit['optimizer_learning_rates_before_step']==[1.8e-5]
assert read(folder/'training_routes.json')['status']=='PASS'
assert read(folder/'actual_training_entry.json')['matmul_allow_tf32']==(args.precision=='tf32')
execution=read(folder/'upstream_merge_training_execution.json')
assert execution['status']=='COMPLETE' and execution['optimizer_steps']==2 and execution['registration_observation']['actual_cuda_ATen_calls']
ref_manifest=read(HERE/'current_bmm_policy_existing_H100_snapshot_manifest.json')[args.precision]
assert file_sha(h/'first_step_local_snapshot.pt')==ref_manifest['sha256']
actual,old,reference=[torch.load(d/'first_step_local_snapshot.pt',map_location='cpu',mmap=True,weights_only=False) for d in (folder,base,h)]
assert actual['initial_sha256']==old['initial_sha256']==reference['initial_sha256'] and len(actual['initial_sha256'])==4174
assert len(actual['gradients'])==len(old['gradients'])==len(reference['gradients'])==4170
assert len(actual['updates'])==len(old['updates'])==len(reference['updates'])==4174
assert execution['registration_observation']['gradient_or_kernel_replacement_by_observer'] is False
assert all('/packages/merged_layout_fix/flag_gems/' in x['path'] for x in execution['registration_observation']['public_sources'].values())
result={'status':'RUNNING','precision':args.precision,'actual_training_steps':2,'new_H100_execution':False,'uniform_RNG':random[0],
        'actual_batches_assets_initial_weights_learning_rate':'PASS','actual_learning_rate_step1':1.8e-5,
        'new_losses':reports[0]['losses'],'old_PPU_losses':reports[1]['losses'],'H100_losses':reports[2]['losses'],
        'all_loss_components_equal_old_PPU':reports[0]['losses']==reports[1]['losses'],'execution':execution,
        'snapshots':{name:{'sha256':file_sha(d/'first_step_local_snapshot.pt'),'bytes':(d/'first_step_local_snapshot.pt').stat().st_size} for name,d in [('candidate',folder),('old_PPU',base),('H100',h)]},'metrics':{}}
for key in ('gradients','updates'):
    entry={}
    for label,left,right in [('new_vs_H100',actual[key],reference[key]),('old_vs_H100',old[key],reference[key]),('new_vs_old_PPU',actual[key],old[key])]:
        entry[label]=compact(metrics(left,right));print('COMPLETE_FULL_VECTOR',key,label,entry[label]['total']['relative_l2'],flush=True)
    result['metrics'][key]=entry
    (args.output_root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
assert result['metrics']['gradients']['new_vs_H100']['total']['elements']==464441841
zero,oldzero=[torch.load(d/'post_step0_adam_state.pt',map_location='cpu',mmap=True,weights_only=False) for d in (folder,base)]
assert zero['optimizer_parameter_names']==oldzero['optimizer_parameter_names'] and zero['optimizer']['param_groups']==oldzero['optimizer']['param_groups']
assert zero['learning_rates']==[0.0] and all(float(v['step'])==1. for v in zero['optimizer']['state'].values())
result['step0_state_changes']={key:compact(metrics(moments(zero,key),moments(oldzero,key))) for key in ('exp_avg','exp_avg_sq')}
result['status']='COMPLETE_FULL_TRAINING_COMPARISON';assert not torch.cuda.is_initialized()
(args.output_root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(result['status'],flush=True)
