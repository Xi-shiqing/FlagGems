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
for candidate in fixed_rne_once fixed_rne_bits_once fixed_rna_bits_once; do
  "$python" "$trial/check_triangle_key_positions.py" --server h100 --candidate "$candidate" --output "$trial/key_positions_$candidate.json"
done
