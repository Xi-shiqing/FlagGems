#!/usr/bin/env bash
set -uo pipefail
work=/mnt/cpfs/xishiqing/ppu_tf32_rne_20260923/host_pipeline_20261004
while [[ ! -f $work/screen_v2.exit ]]; do sleep 15; done
if [[ $(cat "$work/screen_v2.exit") != 0 ]]; then echo 99 > "$work/matrix.exit"; exit 99; fi
mode=$(python3 - <<'SELECT'
import json
from pathlib import Path
p=Path('/mnt/cpfs/xishiqing/ppu_tf32_rne_20260923/host_pipeline_20261004/screen_v2/G01/protenix_base_default_v1.0.0/paired.json')
r=json.loads(p.read_text())['summary']
a,b,c=[r[m]['median_seconds'] for m in ('tf32','tf32_fullwrite','tf32_layout')]
mode='tf32_layout' if c<a*.99 and c<b else 'tf32_fullwrite'
Path('/mnt/cpfs/xishiqing/ppu_tf32_rne_20260923/host_pipeline_20261004/candidate_selection.json').write_text(json.dumps({'mode':mode,'screen_G01_times':{m:v['median_seconds'] for m,v in r.items()},'selection':'Prefer combined RHS preparation only if long-screen time improves by >1% and beats allocation alone; final nine-model acceptance remains required.'},indent=2)+'\n')
print(mode)
SELECT
)
export HOST_PIPELINE_CANDIDATE_MODE=$mode
cp "$work/candidate_v2/flag_gems/ops/mm_preserve_layout.py" "$work/mm_preserve_layout_screen_version.py"
cp "$work/mm_preserve_layout_fast_fallback.py" "$work/mm_preserve_layout.py"
cp "$work/mm_preserve_layout.py" "$work/candidate_v2/flag_gems/ops/mm_preserve_layout.py"
bash "$work/run_matrix_ppu.sh"
result=$?
echo "$result" > "$work/matrix.exit"
if [[ $result != 0 ]]; then exit $result; fi
bash "$work/run_profile_candidate_ppu.sh"
result=$?
echo "$result" > "$work/profile_candidate.exit"
exit "$result"
