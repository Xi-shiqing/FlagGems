"""Exclusive idle-card admission; original model and unified RNG environment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OLD = Path('/mnt/cpfs/xishiqing/ppu_full_acceptance_20261005/precision_residual_20261019')
L = Path('/root/xishiqing_protenix_precision_20261023')
sys.path.insert(0, str(OLD))
from ppu_artifact_volume_gate_20261023 import acquire

parser=argparse.ArgumentParser();parser.add_argument('--precision',choices=['fp32','tf32'],required=True);args=parser.parse_args()
check=json.loads((HERE/'precision_routes_layout.json').read_text())
assert check['status']=='PASS_ALL_CURRENT_PRECISION_ROUTES'
assert json.loads((HERE/'layout_validation_complete.json').read_text())['status']=='PASS'
output=L/('upstream_merge_training_'+args.precision+'_20261106_layout_fix');output.mkdir(exist_ok=False)
source=(OLD/'linear_x3_environment_20261023.sh').read_text()
changes={
    'entry=$task_here/train_linear_x3_20261023.py': 'entry='+str(HERE/'train_upstream_merge_layout.py'),
    'export ACCEPTANCE_GEMS_PATH=$task_here/gems_canonical_softmax_v2':
        'export ACCEPTANCE_GEMS_PATH='+str(HERE/'packages/merged_layout_fix'),
    'export PYTHONPATH=$task_here/model_ieee_matching:$PYTHONPATH':
        'export PYTHONPATH='+str(HERE)+':$task_here/ppu229_inference_startup_20261023:$task_here/model_ieee_matching:$PYTHONPATH',
    'linear_x3_20261023_${precision}':'upstream_merge_training_20261106_${precision}',
}
for old,new in changes.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
(output/'actual_runtime.sh').write_text(source)
lock,reason,card=acquire('upstream_merge_training_'+args.precision+'_20261106_layout_fix',output/'admission.json','exclusive',preferred=3 if args.precision=='fp32' else 4)
try:
    (output/'card.json').write_text(json.dumps(card,indent=2)+'\n')
    command=['docker','exec','-e','PROTENIX_UPSTREAM_MERGE_RULE=accurate',
             'xishiqing-protenix-precision-runtime-20261023','bash','--noprofile','--norc','-c',source,
             'upstream_merge_training','train',args.precision,str(card['gpu']),str(output),'template']
    with (output/'run.log').open('xb') as stream:
        code=subprocess.call(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    (output/'exit_code').write_text(str(code));assert code==0,'Preserve failed training'
finally:
    if lock.exists() and lock.read_text()==reason:
        lock.unlink()
