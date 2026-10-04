#!/usr/bin/env bash
set -euo pipefail
worker=${1:?worker}; export CUDA_VISIBLE_DEVICES=${2:?gpu}
root=/data/xishiqing
cross=$root/ppu_mm_crossdevice_20260920
vendor=$cross/vendor_gemm_20260930
work=$cross/confidence_precision_20261004
repo=$root/protenix-flagos-port/workspace/Protenix-FlagOS
python=$root/yes-star24-home/envs/protenix-baseline/bin/python
export PYTHONPATH=$work:$repo/integrations/validation/cross_device_rng:$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PROTENIX_CROSS_DEVICE_RNG=1 PROTENIX_RNG_AUDIT=1 PROTENIX_RNG_SEED=101
export PROTENIX_ROOT_DIR=$root/yes-star24-home/projects/protenix-baseline
export MM_SHARED_ESM_ROOT=$cross/shared_assets LAYERNORM_TYPE=fast_layernorm CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
models=(protenix_base_default_v0.5.0 protenix-v2 protenix_mini_esm_v0.5.0 protenix_base_default_v1.0.0 protenix_base_20250630_v1.0.0 protenix_base_constraint_v0.5.0 protenix_mini_default_v0.5.0 protenix_tiny_default_v0.5.0 protenix_mini_ism_v0.5.0)
cd "$repo"
failures=0
for ((i=worker;i<${#models[@]};i+=2)); do
 model=${models[$i]}
 for case_id in G01 G02; do
  input=$root/ppu_mm_matrix_20260919/inputs/$case_id/input.json
  if [[ -f $cross/inputs/$case_id/input.json ]]; then input=$cross/inputs/$case_id/input.json; fi
  out=$work/matrix/$case_id/$model
  mkdir -p "$out"
  export PROTENIX_RNG_TRACE=$out/rng_trace.jsonl
  set +e
  timeout 3600 "$python" "$vendor/audit_model_run.py" --harness "$work/precision_entry.py" --audit-helper "$vendor/audit_model_inputs.py" \
   --boundary-harness "$vendor/benchmark_e2e.py" --server h100 --reference-pair "$work/reference/$case_id/$model/paired.json" --paired-output "$out/paired.json" --per-mode-warmup 1 --per-mode-repeats 3 \
   --repo-root "$repo" --input "$input" --out-root "$out" --model "$model" --use-msa true --use-template false --use-rna-msa false \
   --dtype fp32 --seed 101 --cycle 1 --step 1 --sample 1 --triatt-kernel cuequivariance --trimul-kernel torch \
   --enable-cache true --enable-fusion true --enable-tf32 true --warmup 6 --repeats 18 --save-predictions > "$out/run.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$out/exit_code"
  echo "CLOSURE_MATRIX $case_id $model gpu=$CUDA_VISIBLE_DEVICES exit=$rc"
  if [[ $rc != 0 ]]; then failures=$((failures+1)); fi
 done
done
echo "$failures" > "$work/matrix_worker_$worker.failures"
[[ $failures == 0 ]]
