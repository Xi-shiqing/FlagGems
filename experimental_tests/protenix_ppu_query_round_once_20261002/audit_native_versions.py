"""Record isolated native imports without inspecting native implementation."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CODE = '''import torch,json
from importlib import metadata
import cuequivariance_ops_torch as cue
print(json.dumps({'torch':torch.__version__, 'torch_package':metadata.version('cuequivariance-ops-torch-cu12'), 'ops_package':metadata.version('cuequivariance-ops-cu12'), 'import_path':cue.__file__}))
'''
rows = {}
for name, paths in (('frozen', str(ROOT)), ('isolated_latest', str(ROOT / 'vendor_0120') + ':' + str(ROOT))):
    env = os.environ.copy()
    env['PYTHONPATH'] = paths
    value = subprocess.run([sys.executable, '-c', CODE], env=env, check=True, text=True, capture_output=True)
    rows[name] = json.loads(value.stdout.strip().splitlines()[-1])
assert rows['frozen']['ops_package'] == rows['frozen']['torch_package'] == '0.8.0'
assert rows['isolated_latest']['ops_package'] == rows['isolated_latest']['torch_package'] == '0.12.0'
(ROOT / 'native_versions.json').write_text(json.dumps(rows, indent=2) + '\n')
print('NATIVE_VERSION_IMPORT_AUDIT_PASS')
