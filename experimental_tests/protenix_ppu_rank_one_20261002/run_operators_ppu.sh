#!/usr/bin/env bash
set -euo pipefail
root=/mnt/cpfs/xishiqing
exp=$root/ppu_tf32_rne_20260923
source "$exp/vendor_gemm_20260930/env_ppu.sh"
work=$exp/rank_one_20261002
export PYTHONPATH=$work/candidate:$work:$exp:$repo
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$repo"
cp "$work/rank_one.py" "$work/candidate/flag_gems/ops/rank_one.py"
python3 "$work/check_rank_one.py" --out "$work/check.json" > "$work/check.log" 2>&1
python3 "$work/bench_rank_one.py" --server ppu --baseline "$exp/jittor_reference_20261001/candidate" --output "$work/operators.json"
