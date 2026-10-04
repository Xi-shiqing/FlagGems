#!/usr/bin/env bash
set -euo pipefail
root=/mnt/cpfs/xishiqing
exp=$root/ppu_tf32_rne_20260923
source "$exp/vendor_gemm_20260930/env_ppu.sh"
work=$exp/closure_20261003
training=$root/protenix-training-parity-20260825
python=$(command -v python)
checkpoint=$root/root_migrated/checkpoint/protenix-v2.pt
common=$root/root_migrated/common
export PROTENIX_EXTRA_SITE_PACKAGES=$root/ppu224_container_mount/site-packages
export PYTHONPATH=$work:$exp/gemv_qk_20261002/column_candidate:$exp/matched_tf32_20261001:$exp/vendor_gemm_20260930:$root/shared_rng:$training/scripts:$repo
export PROTENIX_TRIATTENTION_FLAGGEMS_PUBLIC=1 PROTENIX_TRIATTENTION_FLAGGEMS=1
export PROTENIX_TRIATTENTION_PPU_SMALL=0 PROTENIX_TRIATTENTION_PPU_STREAMING=0
export CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=20260825
export PROTENIX_CROSS_DEVICE_RNG=1 PROTENIX_RNG_AUDIT=1 PROTENIX_RNG_SEED=20260825 PROTENIX_PATCH_GEOMETRY_RNG=1
export PROTENIX_DISABLE_PERMUTATION_ERROR_DUMP=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROTENIX_TF32_RESTORE_MODULES="" FLAG_GEMS_TRIANGLE_ATTENTION_FP16_TRAINING_SCRATCH=0
cd "$repo"
failures=0
for crop in 245; do
 for mode in backward_only_tf32; do
  out=$work/training_confirm/$crop/$mode
  mkdir -p "$out"
  export PROTENIX_RNG_TRACE=$out/rng_trace.jsonl
  precision=fp32; candidate=public
  if [[ $mode != fp32 ]]; then precision=tf32; fi
  extra=()
  if [[ $mode == backward_only_tf32 ]]; then candidate=backward_only;  fi
  export PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION=ieee
  if [[ $precision == tf32 ]]; then export PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION=tf32; fi
  set +e
  timeout 3600 "$python" "$work/train_entry.py" --existing-tool "$training/scripts/training_parity_benchmark_20260825.py" \
   --precision "$precision" --candidate "$candidate" --skip-parameter-audit "${extra[@]}" --platform ppu810e --model protenix-v2 \
   --checkpoint "$checkpoint" --fixture-root "$training/fixture" --common-root "$common" --output-root "$out" \
   --seed 20260825 --max-steps 4 --crop-size "$crop" --dtype fp32 --triangle-attention triattention --triangle-multiplicative torch \
   --blocks-per-ckpt 1 --pairformer-blocks 48 --msa-blocks 4 --diffusion-batch-size 1 --load-strict --shared-rng \
   --no-track-parameter-updates --run-tag closure_${crop}_${mode} > "$out/run.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$out/exit_code"
  echo "FULL_TRAINING $crop $mode exit=$rc"
  if [[ $rc != 0 ]]; then failures=$((failures+1)); fi
 done
done
echo "$failures" > "$work/backward_only_confirm.failures"
[[ $failures == 0 ]]
