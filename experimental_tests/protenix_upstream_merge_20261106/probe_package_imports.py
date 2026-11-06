"""Probe complete source packages without installing or replacing dependencies."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OLD = Path('/mnt/cpfs/xishiqing/ppu_full_acceptance_20261005/precision_residual_20261019')
sys.path.insert(0, str(OLD))
from ppu_artifact_volume_gate_20261023 import acquire

variants = sys.argv[1:] or ['ours', 'upstream']
lock, reason, card = acquire('upstream_package_imports_20261106', HERE/'import_admission.json', 'exclusive', preferred=2)
probe = '''import sys,json,hashlib
from pathlib import Path
sys.meta_path[:]=[x for x in sys.meta_path if '_flag_gems_editable' not in getattr(x,'__module__','')]
import torch,triton,flag_gems
r={'status':'PASS_COMPLETE_PACKAGE_IMPORT','torch':torch.__version__,'triton':triton.__version__,'package':flag_gems.__file__,'vendor':flag_gems.vendor_name,'device':torch.cuda.get_device_name(),'functions':{}}
for name in ['mm','bmm','addmm','linear','index_select_backward','pad','mean','layer_norm_backward','triangle_attention']:
 f=getattr(flag_gems,name,None)
 if f is None:r['functions'][name]=None;continue
 p=Path(f.__code__.co_filename)
 r['functions'][name]={'module':f.__module__,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps(r))
'''
results = {}
try:
    for variant in variants:
        package = HERE/'packages'/variant
        shell = ('set -euo pipefail\nsource /mnt/cpfs/xishiqing/ppu_tf32_rne_20260923/vendor_gemm_20260930/env_ppu.sh\n'
                 + 'export CUDA_VISIBLE_DEVICES='+str(card['gpu'])+' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n'
                 + 'export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(package))+'\n'
                 + 'exec python3 -c '+shlex.quote(probe))
        run = subprocess.run(['docker','exec','xishiqing-protenix-precision-runtime-20261023','bash','--noprofile','--norc','-c',shell], capture_output=True,text=True)
        record = {'returncode':run.returncode,'stdout':run.stdout[-2000:],'stderr':run.stderr[-5000:]}
        if run.returncode == 0:
            record['result'] = json.loads(run.stdout.splitlines()[-1])
        results[variant] = record
        (HERE/('import_probe_'+variant+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(variant,run.returncode,flush=True)
finally:
    if lock.exists() and lock.read_text() == reason:
        lock.unlink()
