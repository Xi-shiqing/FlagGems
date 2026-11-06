"""Run the merged repository's relevant pytest tests in the unchanged container."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OLD = Path('/mnt/cpfs/xishiqing/ppu_full_acceptance_20261005/precision_residual_20261019')
sys.path.insert(0, str(OLD))
from ppu_artifact_volume_gate_20261023 import acquire

suite = HERE/'validation_suite_ready'
output = HERE/'validation_results_ready'
output.mkdir(exist_ok=False)
lock, reason, card = acquire('upstream_merged_pytest_ready_20261106', output/'admission.json', 'exclusive', preferred=2)
files = ['copy', 'div', 'mean', 'linear', 'pad', 'sigmoid', 'silu', 'unfold_backward',
         'log_softmax', 'index_put', 'index_select', 'layer_norm', 'softmax', 'narrow', 'sum',
         'scaled_dot_product_attention_math', 'square', 'libentry', 'registration_contract']
entry = '''import sys,signal
sys.meta_path[:]=[x for x in sys.meta_path if '_flag_gems_editable' not in getattr(x,'__module__','')]
def timeout(*args):raise TimeoutError('Bound individual validation file at 600 seconds')
signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
'''
results = []
try:
    for name in files:
        command = ['python3','-c',entry,'tests/test_'+name+'.py','--quick','--ref','cpu',
                   '--first-parameter-only','--maxfail=1','-q','--junitxml='+str(output/(name+'.xml'))]
        if name == 'square':
            command.remove('--first-parameter-only')
            command[3] += '::test_square_preserves_native_autograd'
        if name == 'libentry':
            command.remove('--maxfail=1')
        shell = ('set -euo pipefail\nsource /mnt/cpfs/xishiqing/ppu_tf32_rne_20260923/vendor_gemm_20260930/env_ppu.sh\n'
                 + 'export CUDA_VISIBLE_DEVICES='+str(card['gpu'])+' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n'
                 + 'export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+str(HERE/'packages/merged_ready')+':'+str(suite)+'\n'
                 + 'export TRITON_CACHE_DIR='+str(HERE/'triton_cache')+' FLAGGEMS_CACHE_DIR='+str(HERE/'validation_gems_cache')+'\n'
                 + 'cd '+shlex.quote(str(suite))+'\nexec '+shlex.join(command))
        (output/(name+'_runtime.sh')).write_text(shell)
        with (output/(name+'.log')).open('xb') as log:
            code = subprocess.call(['docker','exec','xishiqing-protenix-precision-runtime-20261023','bash','--noprofile','--norc','-c',shell],stdout=log,stderr=subprocess.STDOUT)
        row = {'file':name,'returncode':code}
        results.append(row)
        (output/'status.json').write_text(json.dumps(results,indent=2)+'\n')
        print('VALIDATION_FILE',name,code,flush=True)
finally:
    if lock.exists() and lock.read_text() == reason:
        lock.unlink()

assert len(results)==len(files) and all(row["returncode"]==0 for row in results), "Inspect preserved per-file failures"
