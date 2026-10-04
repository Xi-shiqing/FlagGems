#!/usr/bin/env bash
set -uo pipefail
work=/mnt/cpfs/xishiqing/ppu_tf32_rne_20260923/native_reference_e2e_20261004
bash "$work/run_ppu.sh" 0 0 > "$work/worker_0.log" 2>&1 &
pid0=$!
bash "$work/run_ppu.sh" 1 7 > "$work/worker_1.log" 2>&1 &
pid1=$!
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
echo "$rc0 $rc1" > "$work/matrix.exit"
[[ $rc0 == 0 && $rc1 == 0 ]]
