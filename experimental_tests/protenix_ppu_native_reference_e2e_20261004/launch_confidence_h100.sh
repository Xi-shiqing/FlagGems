#!/usr/bin/env bash
set -uo pipefail
work=/data/xishiqing/ppu_mm_crossdevice_20260920/native_reference_e2e_20261004
python=/data/xishiqing/yes-star24-home/envs/protenix-baseline/bin/python
[[ $(cat "$work/matrix.exit") == '0 0' ]] || exit 1
bash "$work/run_confidence_h100.sh" 0 0 > "$work/confidence_worker_0.log" 2>&1 &
pid0=$!
bash "$work/run_confidence_h100.sh" 1 7 > "$work/confidence_worker_1.log" 2>&1 &
pid1=$!
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
echo "$rc0 $rc1" > "$work/confidence_matrix.exit"
[[ $rc0 == 0 && $rc1 == 0 ]]
