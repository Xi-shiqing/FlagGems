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
lock, reason, card = acquire('upstream_autograd_registration_v2_20261106', HERE/'admission_autograd_registration_v2.json', 'exclusive', preferred=1)
try:
    (HERE/'card_autograd_registration_v2.json').write_text(json.dumps(card, indent=2)+'\n')
    shell = '''set -euo pipefail
source /mnt/cpfs/xishiqing/ppu_tf32_rne_20260923/vendor_gemm_20260930/env_ppu.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="%s"
export PYTHONPATH=/root/xishiqing_protenix_precision_20261023/upstream_merge_20261106/packages/merged_ready
export TRITON_CACHE_DIR=/root/.triton/cache
export FLAGGEMS_CACHE_DIR="%s/gems_cache"
exec python3 "%s/check_autograd_registration_v2.py" --output "%s/autograd_registration_v2_before.json"
''' % (card['gpu'], HERE, HERE, HERE)
    (HERE/'actual_runtime_autograd_registration_v2.sh').write_text(shell)
    code = subprocess.call(['docker','exec',container,'bash','--noprofile','--norc','-c',shell])
    assert code == 0, 'Preserve failed comparison'
finally:
    if lock.exists() and lock.read_text() == reason:
        lock.unlink()
