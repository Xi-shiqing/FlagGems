#!/usr/bin/env bash
set -euo pipefail
root=/mnt/cpfs/xishiqing
exp=$root/ppu_tf32_rne_20260923
source "$exp/vendor_gemm_20260930/env_ppu.sh"
work=$exp/gemv_qk_20261002
export PYTHONPATH=$work/candidate:$work:$exp:$repo
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$repo"
python3 "$work/stage_gemv.py"
python3 "$work/check_gemv.py" > "$work/check.log" 2>&1
python3 "$work/bench_gemv.py" --server ppu --baseline "$exp/rank_one_20261002/candidate" --output "$work/operators.json"
