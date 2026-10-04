#!/usr/bin/env bash
set -euo pipefail
root=/mnt/cpfs/xishiqing
exp=$root/ppu_tf32_rne_20260923
vendor=$exp/vendor_gemm_20260930
source "$vendor/env_ppu.sh"
work=$exp/gemv_qk_20261002
export PYTHONPATH=$exp/rank_one_20261002/candidate:$work:$vendor:$root/shared_rng:$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PROTENIX_CROSS_DEVICE_RNG=1 PROTENIX_RNG_AUDIT=1 PROTENIX_RNG_SEED=101
export PROTENIX_TRIATTENTION_FLAGGEMS_PUBLIC=1 PROTENIX_TRIATTENTION_FLAGGEMS=0
export PROTENIX_TRIATTENTION_PPU_STREAMING=0 PROTENIX_TRIATTENTION_PPU_SMALL=0
export PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION=tf32 MM_PROFILE=0 MM_DIAGNOSTIC_TRACE=0
export MM_SHARED_ESM_ROOT=$root/ppu_mm_crossdevice_20260920/shared_assets
out=$work/qk/G02
mkdir -p "$out"
export PROTENIX_RNG_TRACE=$out/rng_trace.jsonl
cd "$repo"
timeout 2400 python3 "$vendor/legacy_msa_harness.py" \
 --entrypoint "$vendor/audit_gems_entrypoints.py" --numpy-root "$vendor/numpy126" \
 --work-root "$out/msa_worker" --gems-entrypoint "$work/replay_model.py" --entry-audit "$out/gems_entry_audit.json" \
 --replay-harness "$exp/generic_pipeline_20260926/benchmark_model.py" --replay-output "$out/replay.json" \
 --reference-pair "$exp/rank_one_20261002/model/G02/paired.json" --reference-mode candidate_tf32 \
 --repo-root "$repo" --input "$root/ppu_mm_generalization_20260919/inputs/G02/input.json" \
 --out-root "$out" --model protenix_base_default_v1.0.0 --use-msa true --use-template false --use-rna-msa false \
 --dtype fp32 --seed 101 --cycle 1 --step 1 --sample 1 --triatt-kernel triattention --trimul-kernel torch \
 --enable-cache true --enable-fusion true --enable-tf32 true --warmup 1 --repeats 1 --use-flag-gems --save-predictions
