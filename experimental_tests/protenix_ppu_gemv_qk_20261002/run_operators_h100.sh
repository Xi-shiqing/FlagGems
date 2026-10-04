#!/usr/bin/env bash
set -euo pipefail
work=/data/xishiqing/ppu_mm_crossdevice_20260920/gemv_qk_20261002
repo=/data/xishiqing/protenix-flagos-port/workspace/Protenix-FlagOS
export CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$work:$repo
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$repo"
/data/xishiqing/yes-star24-home/envs/protenix-baseline/bin/python "$work/bench_gemv.py" --server h100 --output "$work/operators.json"
