#!/usr/bin/env bash
set -euo pipefail
root=/mnt/cpfs/xishiqing
exp=$root/ppu_tf32_rne_20260923
vendor=$exp/vendor_gemm_20260930
source "$vendor/env_ppu.sh"
export CUDA_VISIBLE_DEVICES=7
work=$exp/host_pipeline_20261004
prior=$exp/confidence_precision_20261004
export PYTHONPATH=$work/candidate_v2:$work:$prior:$exp/matched_tf32_20261001:$vendor:$root/shared_rng:$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PROTENIX_CROSS_DEVICE_RNG=1 PROTENIX_RNG_AUDIT=1 PROTENIX_RNG_SEED=101
export PROTENIX_TRIATTENTION_FLAGGEMS_PUBLIC=1 PROTENIX_TRIATTENTION_FLAGGEMS=0 PROTENIX_TRIATTENTION_PPU_STREAMING=0 PROTENIX_TRIATTENTION_PPU_SMALL=0
export MM_PROFILE=0 MM_DIAGNOSTIC_TRACE=0 MM_SHARED_ESM_ROOT=$root/ppu_mm_crossdevice_20260920/shared_assets
candidate_mode=${HOST_PIPELINE_CANDIDATE_MODE:-tf32_layout}
cd "$repo"
models=(protenix_base_default_v1.0.0)
for model in "${models[@]}"; do
for case_id in G02; do
 out=$work/recheck/$case_id/$model
 mkdir -p "$out"
 export PROTENIX_RNG_TRACE=$out/rng_trace.jsonl PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION=tf32
 timeout 3600 python3 "$vendor/legacy_msa_harness.py" --entrypoint "$vendor/audit_gems_entrypoints.py" --numpy-root "$vendor/numpy126" \
  --work-root "$out/msa_worker" --gems-entrypoint "$work/matrix_entry.py" --entry-audit "$out/gems_entry_audit.json" \
  --boundary-harness "$exp/generic_pipeline_20260926/benchmark_model.py" --server ppu --reference-pair "$prior/matrix/$case_id/$model/paired.json" --paired-output "$out/paired.json" \
  --variants fp32 tf32 "$candidate_mode" --per-mode-warmup 3 --per-mode-repeats 15 --poison \
  --repo-root "$repo" --input "$root/ppu_mm_generalization_20260919/inputs/$case_id/input.json" \
  --out-root "$out" --model "$model" --use-msa true --use-template false --use-rna-msa false --dtype fp32 --seed 101 --cycle 1 --step 1 --sample 1 \
  --triatt-kernel triattention --trimul-kernel torch --enable-cache true --enable-fusion true --enable-tf32 true \
  --warmup 9 --repeats 45 --use-flag-gems --save-predictions > "$out/run.log" 2>&1
 echo "MATRIX_PASS $case_id $model"
done

done
