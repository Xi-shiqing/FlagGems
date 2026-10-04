#!/usr/bin/env bash
set -uo pipefail
work=/data/xishiqing/ppu_mm_crossdevice_20260920/confidence_precision_20261004
while [[ ! -f "$work/matrix.exit" ]]; do sleep 10; done
[[ $(cat "$work/matrix.exit" | tr -d " 0\n") == "" ]] || exit 1
python3 - <<'REF'
from pathlib import Path
import json
root=Path('/data/xishiqing/ppu_mm_crossdevice_20260920/confidence_precision_20261004')
prior=root.parent/'native_reference_e2e_20261004'
for run in (prior/'corrected_matrix').glob('*/*'):
 p=json.loads((prior/'matrix'/run.parent.name/run.name/'paired.json').read_text())
 p['summary']['tf32_triangle_corrected']=json.loads((run/'paired.json').read_text())['summary']['tf32_triangle_corrected']
 out=root/'reference'/run.parent.name/run.name/'paired.json'
 out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(p)+'\n')
REF
[[ $? == 0 ]] || exit 1
bash "$work/run_fused_h100.sh" 0 0 > "$work/fused_worker_0.log" 2>&1 &
pid0=$!
bash "$work/run_fused_h100.sh" 1 7 > "$work/fused_worker_1.log" 2>&1 &
pid1=$!
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
echo "$rc0 $rc1" > "$work/fused_matrix.exit"
[[ $rc0 == 0 && $rc1 == 0 ]]
