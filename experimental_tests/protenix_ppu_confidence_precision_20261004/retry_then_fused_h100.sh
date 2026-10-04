#!/usr/bin/env bash
set -euo pipefail
work=/data/xishiqing/ppu_mm_crossdevice_20260920/confidence_precision_20261004
mkdir -p "$work/control_failures"
cp "$work/matrix.exit" "$work/control_failures/matrix_firstpass.exit"
mv "$work/matrix/G01/protenix_mini_ism_v0.5.0" "$work/control_failures/mini_ism_esm_preprocess_collision"
bash "$work/retry_h100.sh" 0 0
python3 - <<'AUDIT'
from pathlib import Path
root=Path('/data/xishiqing/ppu_mm_crossdevice_20260920/confidence_precision_20261004')
markers=list((root/'matrix').glob('*/*/exit_code'))
assert len(markers)==18 and all(p.read_text().strip()=='0' for p in markers)
(root/'matrix.exit').write_text('0 0\n')
AUDIT
bash "$work/launch_fused_h100.sh"
