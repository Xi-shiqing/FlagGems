#!/usr/bin/env bash
set -euo pipefail
root=/data/xishiqing/ppu_mm_crossdevice_20260920
trial="$root/query_round_once_20261002"
repo=/data/xishiqing/protenix-flagos-port/workspace/Protenix-FlagOS
python=/data/xishiqing/yes-star24-home/envs/protenix-baseline/bin/python
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="$trial:$repo"
unset PROTENIX_TF32_CAPTURE_DIR
export PROTENIX_TF32_RESTORE_MODULES=""
"$python" "$trial/check_rounding_bits.py" --server h100 --output "$trial/rounding_bits.json"
"$python" "$trial/bench_query_once.py" --server h100 --output "$trial/triangle_training.json" --profile
