#!/usr/bin/env bash
set -euo pipefail
root=/mnt/cpfs/xishiqing
exp="$root/ppu_tf32_rne_20260923"
trial="$exp/query_round_once_20261002"
source "$exp/vendor_gemm_20260930/env_ppu.sh"
repo="$root/flagtree_ppu_docker_20260821/experiments/public_e2e_profile_20260822/Protenix-public-profile"
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="$trial:$exp/gemv_qk_20261002/column_candidate:$exp/matched_tf32_20261001:$exp/vendor_gemm_20260930:$root/shared_rng:$repo"
unset PROTENIX_TF32_CAPTURE_DIR
export PROTENIX_TF32_RESTORE_MODULES=""
for candidate in fixed_rne_once fixed_rne_bits_once fixed_rna_bits_once; do
  python "$trial/check_triangle_key_positions.py" --server ppu --candidate "$candidate" --output "$trial/key_positions_$candidate.json"
done
