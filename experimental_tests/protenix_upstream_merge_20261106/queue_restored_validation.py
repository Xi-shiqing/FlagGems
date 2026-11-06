"""Acquire an idle PPU and compare operator sources in the original container."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OLD = Path('/mnt/cpfs/xishiqing/ppu_full_acceptance_20261005/precision_residual_20261019')
sys.path.insert(0, str(OLD))
from ppu_artifact_volume_gate_20261023 import acquire

container = 'xishiqing-protenix-precision-runtime-20261023'
assert subprocess.check_output(['docker', 'inspect', '-f', '{{.State.Running}} {{.Image}}', container], text=True).strip() == 'true sha256:307ab15433b3f6775d8f204fc9a469e1e8b9989124dd8aea46ea60e9aa6429ec'
lock, reason, card = acquire('upstream_precision_route_layout_20261106', HERE/'admission_precision_restored.json', 'exclusive', preferred=1)
try:
    (HERE/'card_precision_restored.json').write_text(json.dumps(card, indent=2)+'\n')
    shell = '''set -euo pipefail
source /mnt/cpfs/xishiqing/ppu_tf32_rne_20260923/vendor_gemm_20260930/env_ppu.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="%s"
export PYTHONPATH=/root/xishiqing_protenix_precision_20261023/upstream_merge_20261106/packages/merged_restored
export TRITON_CACHE_DIR=/root/.triton/cache
export FLAGGEMS_CACHE_DIR="%s/gems_cache"
exec python3 "%s/check_precision_routes_v3.py" --output "%s/precision_routes_restored.json"
''' % (card['gpu'], HERE, HERE, HERE)
    (HERE/'actual_runtime_precision_restored.sh').write_text(shell)
    code = subprocess.call(['docker','exec',container,'bash','--noprofile','--norc','-c',shell])
    assert code == 0, 'Preserve failed comparison'
    assert json.loads((HERE/'precision_routes_restored.json').read_text())['status']=='PASS_ALL_CURRENT_PRECISION_ROUTES'
    test_shell = shell[:shell.index('exec python3')]+"cd "+str(HERE/'validation_suite_restored')+"\nexec python3 -c "+__import__('shlex').quote("import sys;sys.meta_path[:]=[x for x in sys.meta_path if '_flag_gems_editable' not in getattr(x,'__module__','')];import pytest;raise SystemExit(pytest.main(sys.argv[1:]))")+" tests/test_linear.py::test_linear_backward_noncontiguous tests/test_linear.py::test_linear_layout_and_empty_dimensions tests/test_linear.py::test_linear_backward_empty_dimensions tests/test_cdist.py::test_cdist_registered_backward tests/test_mm.py::test_mm_empty_dimensions_registered --ref cpu -q --junitxml="+str(HERE/'linear_restored.xml')
    with (HERE/'linear_restored.log').open('xb') as log:
        code = subprocess.call(['docker','exec',container,'bash','--noprofile','--norc','-c',test_shell],stdout=log,stderr=subprocess.STDOUT)
    (HERE/'linear_restored.exit').write_text(str(code))
    assert code == 0, 'Preserve noncontiguous-layout regression failure'
    (HERE/'restored_validation.json').write_text(json.dumps({'status':'PASS','precision_cases':10,'linear_layout_and_empty_cases':13,'cdist_backward_cases':7,'mm_empty_cases':6}))
finally:
    if lock.exists() and lock.read_text() == reason:
        lock.unlink()
