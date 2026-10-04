#!/usr/bin/env bash
set -uo pipefail
work=/mnt/cpfs/xishiqing/ppu_tf32_rne_20260923/host_pipeline_20261004
while [[ ! -f $work/recheck.exit ]]; do sleep 15; done
if [[ $(cat "$work/recheck.exit") != 0 ]]; then echo 99 > "$work/recheck_extra.exit"; exit 99; fi
export HOST_PIPELINE_CANDIDATE_MODE=tf32_layout
bash "$work/run_recheck_extra_ppu.sh"
result=$?
echo "$result" > "$work/recheck_extra.exit"
exit "$result"
