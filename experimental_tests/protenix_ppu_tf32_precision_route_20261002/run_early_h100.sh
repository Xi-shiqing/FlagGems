#!/usr/bin/env bash
set -euo pipefail
cross=/data/xishiqing/ppu_mm_crossdevice_20260920
base=/data/xishiqing/ppu_mm_matrix_20260919
repo=/data/xishiqing/protenix-flagos-port/workspace/Protenix-FlagOS
python=/data/xishiqing/yes-star24-home/envs/protenix-baseline/bin/python
vendor=$cross/vendor_gemm_20260930
work=$cross/precision_route_20261002
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$work:$repo/integrations/validation/cross_device_rng:$repo
export PROTENIX_CROSS_DEVICE_RNG=1 PROTENIX_RNG_AUDIT=1 PROTENIX_RNG_SEED=101
export PROTENIX_ROOT_DIR=/data/xishiqing/yes-star24-home/projects/protenix-baseline
export MM_SHARED_ESM_ROOT=$cross/shared_assets
export LAYERNORM_TYPE=fast_layernorm CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$repo"
for case_id in G02; do
  model=protenix_base_default_v1.0.0
  input=$base/inputs/$case_id/input.json
  if [[ -f $cross/inputs/$case_id/input.json ]]; then input=$cross/inputs/$case_id/input.json; fi
  refs=$vendor/e2e/$case_id
  if [[ $case_id == G01 ]]; then refs=$vendor/matrix/$case_id/$model; fi
  out=$work/early_linear/$case_id
  mkdir -p "$out"
  export PROTENIX_RNG_TRACE=$out/rng_trace.jsonl
  set +e
  timeout 2400 "$python" "$vendor/audit_model_run.py" --harness "$work/shadow_linear.py" \
    --audit-helper "$vendor/audit_model_inputs.py" --route-harness "$vendor/benchmark_e2e.py" \
    --route-output "$out/routes.json" \
    --repo-root "$repo" --input "$input" --out-root "$out" --model "$model" \
    --use-msa true --use-template false --use-rna-msa false --dtype fp32 --seed 101 \
    --cycle 1 --step 1 --sample 1 --triatt-kernel cuequivariance --trimul-kernel torch \
    --enable-cache true --enable-fusion true --enable-tf32 true \
    --warmup 1 --repeats 1 --save-predictions > "$out/run.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$out/exit_code"
  echo "MATCHED_H100 $case_id exit=$rc"
  [[ $rc == 0 ]]
done
