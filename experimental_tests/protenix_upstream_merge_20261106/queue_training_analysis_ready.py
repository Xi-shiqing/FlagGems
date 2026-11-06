"""Wait for owned training jobs, then compare server-resident tensors using CPU only."""
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
for precision in ['fp32', 'tf32']:
    folder = HERE.parent/('upstream_merge_training_'+precision+'_20261106_ready')
    exit_file = folder/'exit_code'
    deadline = time.monotonic()+5400
    while not exit_file.exists():
        assert time.monotonic()<deadline, 'Training still running; preserve state for inspection'
        time.sleep(20)
    assert exit_file.read_text().strip()=='0', 'Training failed; inspect preserved receipt'
    receipt = json.loads((folder/'upstream_merge_training_execution.json').read_text())
    assert receipt['status']=='COMPLETE'
    output = HERE/('training_comparison_ready_'+precision)
    command = ['docker','exec','-e','CUDA_VISIBLE_DEVICES=','-e','OMP_NUM_THREADS=4',
               '-e','MKL_NUM_THREADS=4','-e','OPENBLAS_NUM_THREADS=4',
               'xishiqing-protenix-precision-runtime-20261023','python3',
               str(HERE/'analyze_upstream_merge_training_ready.py'),
               '--precision',precision,'--output-root',str(output)]
    with (HERE/('training_analysis_ready_'+precision+'.log')).open('xb') as log:
        code = subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
    (HERE/('training_analysis_ready_'+precision+'.exit')).write_text(str(code))
    assert code==0, 'Keep failed analysis and inspect assertions'
    print('COMPLETE_TRAINING_ANALYSIS',precision,flush=True)
