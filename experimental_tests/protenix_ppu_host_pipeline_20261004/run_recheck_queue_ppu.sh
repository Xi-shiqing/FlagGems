#!/usr/bin/env bash
set -uo pipefail
work=/mnt/cpfs/xishiqing/ppu_tf32_rne_20260923/host_pipeline_20261004
while [[ ! -f $work/profile_candidate.exit ]]; do
 if [[ -f $work/matrix.exit && $(cat "$work/matrix.exit") != 0 ]]; then echo 99 > "$work/recheck.exit"; exit 99; fi
 sleep 15
done
export HOST_PIPELINE_CANDIDATE_MODE=tf32_layout
bash "$work/run_recheck_ppu.sh"
result=$?
echo "$result" > "$work/recheck.exit"
exit "$result"
