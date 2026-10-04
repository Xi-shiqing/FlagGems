#!/usr/bin/env bash
set -uo pipefail
work=/data/xishiqing/ppu_mm_crossdevice_20260920/native_reference_e2e_20261004
python=/data/xishiqing/yes-star24-home/envs/protenix-baseline/bin/python
CUDA_VISIBLE_DEVICES=7 PYTHONPATH="$work" PYTHONDONTWRITEBYTECODE=1 "$python" "$work/verify_adapter.py" --output "$work/adapter_validation.json" > "$work/adapter_validation.log" 2>&1
rc=$?
echo "$rc" > "$work/adapter_validation.exit"
if [[ $rc != 0 ]]; then exit "$rc"; fi
bash "$work/run_h100.sh" 0 0 > "$work/worker_0.log" 2>&1 &
pid0=$!
bash "$work/run_h100.sh" 1 7 > "$work/worker_1.log" 2>&1 &
pid1=$!
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
echo "$rc0 $rc1" > "$work/matrix.exit"
[[ $rc0 == 0 && $rc1 == 0 ]]
