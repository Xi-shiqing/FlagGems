#!/usr/bin/env bash
set -uo pipefail
work=/data/xishiqing/ppu_mm_crossdevice_20260920/native_reference_e2e_20261004
python=/data/xishiqing/yes-star24-home/envs/protenix-baseline/bin/python
[[ $(cat "$work/matrix.exit") == '0 0' ]] || exit 1
CUDA_VISIBLE_DEVICES=7 PYTHONPATH="$work" PYTHONDONTWRITEBYTECODE=1 "$python" "$work/verify_adapter.py" --output "$work/adapter_validation_fixed.json" > "$work/adapter_validation_fixed.log" 2>&1
rc=$?
echo "$rc" > "$work/adapter_validation_fixed.exit"
if [[ $rc != 0 ]]; then exit "$rc"; fi
bash "$work/run_corrected_h100.sh" 0 0 > "$work/corrected_worker_0.log" 2>&1 &
pid0=$!
bash "$work/run_corrected_h100.sh" 1 7 > "$work/corrected_worker_1.log" 2>&1 &
pid1=$!
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
echo "$rc0 $rc1" > "$work/corrected_matrix.exit"
[[ $rc0 == 0 && $rc1 == 0 ]]
