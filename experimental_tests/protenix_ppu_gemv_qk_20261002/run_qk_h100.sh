#!/usr/bin/env bash
set -euo pipefail
cross=/data/xishiqing/ppu_mm_crossdevice_20260920
repo=/data/xishiqing/protenix-flagos-port/workspace/Protenix-FlagOS
python=/data/xishiqing/yes-star24-home/envs/protenix-baseline/bin/python
vendor=$cross/vendor_gemm_20260930
work=$cross/gemv_qk_20261002
export CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$work:$repo/integrations/validation/cross_device_rng:$repo
export PROTENIX_CROSS_DEVICE_RNG=1 PROTENIX_RNG_AUDIT=1 PROTENIX_RNG_SEED=101
export PROTENIX_ROOT_DIR=/data/xishiqing/yes-star24-home/projects/protenix-baseline
export MM_SHARED_ESM_ROOT=$cross/shared_assets
export LAYERNORM_TYPE=fast_layernorm CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=$work/qk/G02
mkdir -p "$out"
export PROTENIX_RNG_TRACE=$out/rng_trace.jsonl
input=$cross/inputs/G02/input.json
if [[ ! -f $input ]]; then input=/data/xishiqing/ppu_mm_matrix_20260919/inputs/G02/input.json; fi
cd "$repo"
timeout 2400 "$python" "$vendor/audit_model_run.py" --harness "$work/replay_model.py" \
 --audit-helper "$vendor/audit_model_inputs.py" --replay-harness "$vendor/benchmark_e2e.py" --replay-output "$out/replay.json" \
 --reference-pair "$cross/rank_one_20261002/model/G02/paired.json" --reference-mode tf32 \
 --repo-root "$repo" --input "$input" --out-root "$out" --model protenix_base_default_v1.0.0 \
 --use-msa true --use-template false --use-rna-msa false --dtype fp32 --seed 101 \
 --cycle 1 --step 1 --sample 1 --triatt-kernel cuequivariance --trimul-kernel torch \
 --enable-cache true --enable-fusion true --enable-tf32 true --warmup 1 --repeats 1 --save-predictions
