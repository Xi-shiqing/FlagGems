#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=7
root=/data/xishiqing
cross=$root/ppu_mm_crossdevice_20260920
vendor=$cross/vendor_gemm_20260930
work=$cross/host_pipeline_20261004
prior=$cross/confidence_precision_20261004
repo=$root/protenix-flagos-port/workspace/Protenix-FlagOS
python=$root/yes-star24-home/envs/protenix-baseline/bin/python
export PYTHONPATH=$work:$prior:$repo/integrations/validation/cross_device_rng:$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PROTENIX_CROSS_DEVICE_RNG=1 PROTENIX_RNG_AUDIT=1 PROTENIX_RNG_SEED=101
export PROTENIX_ROOT_DIR=$root/yes-star24-home/projects/protenix-baseline
export MM_SHARED_ESM_ROOT=$cross/shared_assets LAYERNORM_TYPE=fast_layernorm CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$repo"
models=(protenix_base_20250630_v1.0.0 protenix_base_constraint_v0.5.0 protenix_base_default_v0.5.0 protenix_base_default_v1.0.0 protenix_mini_default_v0.5.0 protenix_mini_esm_v0.5.0 protenix_mini_ism_v0.5.0 protenix_tiny_default_v0.5.0)
for model in "${models[@]}"; do
for case_id in G02 G01; do
 input=$root/ppu_mm_matrix_20260919/inputs/$case_id/input.json
 if [[ -f $cross/inputs/$case_id/input.json ]]; then input=$cross/inputs/$case_id/input.json; fi
 out=$work/steady/$case_id/$model
 mkdir -p "$out"
 export PROTENIX_RNG_TRACE=$out/rng_trace.jsonl
 timeout 3600 "$python" "$vendor/audit_model_run.py" --harness "$prior/serialize_entry.py" --audit-helper "$vendor/audit_model_inputs.py" \
  --serialized-entry "$work/matrix_entry.py" \
  --boundary-harness "$vendor/benchmark_e2e.py" --server h100 --reference-pair "$prior/matrix/$case_id/$model/paired.json" --paired-output "$out/paired.json" \
  --variants fp32 tf32 --per-mode-warmup 3 --per-mode-repeats 7 \
  --repo-root "$repo" --input "$input" --out-root "$out" --model "$model" --use-msa true --use-template false --use-rna-msa false \
  --dtype fp32 --seed 101 --cycle 1 --step 1 --sample 1 --triatt-kernel cuequivariance --trimul-kernel torch \
  --enable-cache true --enable-fusion true --enable-tf32 true --warmup 6 --repeats 14 --save-predictions > "$out/run.log" 2>&1
 echo "MATRIX_PASS $case_id $model"
done

done
