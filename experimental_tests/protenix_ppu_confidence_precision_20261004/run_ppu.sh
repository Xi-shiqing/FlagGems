#!/usr/bin/env bash
set -euo pipefail
worker=${1:?worker}
selected_gpu=${2:?gpu}
root=/mnt/cpfs/xishiqing
exp=$root/ppu_tf32_rne_20260923
vendor=$exp/vendor_gemm_20260930
source "$vendor/env_ppu.sh"
export CUDA_VISIBLE_DEVICES=$selected_gpu
work=$exp/confidence_precision_20261004
export PYTHONPATH=$exp/gemv_qk_20261002/column_candidate:$work:$exp/matched_tf32_20261001:$vendor:$root/shared_rng:$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PROTENIX_CROSS_DEVICE_RNG=1 PROTENIX_RNG_AUDIT=1 PROTENIX_RNG_SEED=101
export PROTENIX_TRIATTENTION_FLAGGEMS_PUBLIC=1 PROTENIX_TRIATTENTION_FLAGGEMS=0 PROTENIX_TRIATTENTION_PPU_STREAMING=0 PROTENIX_TRIATTENTION_PPU_SMALL=0
export MM_PROFILE=0 MM_DIAGNOSTIC_TRACE=0 MM_SHARED_ESM_ROOT=$root/ppu_mm_crossdevice_20260920/shared_assets
models=(protenix_base_default_v0.5.0 protenix-v2 protenix_mini_esm_v0.5.0 protenix_base_default_v1.0.0 protenix_base_20250630_v1.0.0 protenix_base_constraint_v0.5.0 protenix_mini_default_v0.5.0 protenix_tiny_default_v0.5.0 protenix_mini_ism_v0.5.0)
cd "$repo"
failures=0
for ((i=worker;i<${#models[@]};i+=4)); do
 model=${models[$i]}
 for case_id in G01 G02; do
  out=$work/matrix/$case_id/$model
  mkdir -p "$out"
  export PROTENIX_RNG_TRACE=$out/rng_trace.jsonl PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION=tf32
  set +e
  timeout 3600 python3 "$vendor/legacy_msa_harness.py" --entrypoint "$vendor/audit_gems_entrypoints.py" --numpy-root "$vendor/numpy126" \
   --work-root "$out/msa_worker" --gems-entrypoint "$work/precision_entry.py" --entry-audit "$out/gems_entry_audit.json" \
   --boundary-harness "$exp/generic_pipeline_20260926/benchmark_model.py" --server ppu --reference-pair "$exp/native_reference_e2e_20261004/matrix/$case_id/$model/paired.json" --paired-output "$out/paired.json" \
   --per-mode-warmup 1 --per-mode-repeats 3 --repo-root "$repo" --input "$root/ppu_mm_generalization_20260919/inputs/$case_id/input.json" \
   --out-root "$out" --model "$model" --use-msa true --use-template false --use-rna-msa false --dtype fp32 --seed 101 --cycle 1 --step 1 --sample 1 \
   --triatt-kernel triattention --trimul-kernel torch --enable-cache true --enable-fusion true --enable-tf32 true \
   --warmup 5 --repeats 15 --use-flag-gems --save-predictions > "$out/run.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$out/exit_code"
  echo "CLOSURE_MATRIX $case_id $model gpu=$CUDA_VISIBLE_DEVICES exit=$rc"
  if [[ $rc != 0 ]]; then failures=$((failures+1)); fi
 done
done
echo "$failures" > "$work/matrix_worker_$worker.failures"
[[ $failures == 0 ]]
