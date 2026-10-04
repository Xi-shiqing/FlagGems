#!/usr/bin/env bash
set -uo pipefail
work=/mnt/cpfs/xishiqing/ppu_tf32_rne_20260923/confidence_precision_20261004
while [[ ! -f "$work/matrix.exit" ]]; do sleep 10; done
[[ $(cat "$work/matrix.exit" | tr -d " 0\n") == "" ]] || exit 1
bash "$work/run_fused_ppu.sh" 0 0 > "$work/fused_worker_0.log" 2>&1 &
pid0=$!
bash "$work/run_fused_ppu.sh" 1 7 > "$work/fused_worker_1.log" 2>&1 &
pid1=$!
bash "$work/run_fused_ppu.sh" 2 2 > "$work/fused_worker_2.log" 2>&1 &
pid2=$!
bash "$work/run_fused_ppu.sh" 3 6 > "$work/fused_worker_3.log" 2>&1 &
pid3=$!
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
wait "$pid2"; rc2=$?
wait "$pid3"; rc3=$?
echo "$rc0 $rc1 $rc2 $rc3" > "$work/fused_matrix.exit"
[[ $rc0 == 0 && $rc1 == 0 && $rc2 == 0 && $rc3 == 0 ]]
