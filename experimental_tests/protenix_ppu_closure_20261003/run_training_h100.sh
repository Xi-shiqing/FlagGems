#!/usr/bin/env bash
set -euo pipefail
root=/data/xishiqing
cross=$root/ppu_mm_crossdevice_20260920
work=$cross/closure_20261003
repo=$root/protenix-flagos-port/workspace/Protenix-FlagOS
training=$root/protenix-training-parity-20260825
python=$root/yes-star24-home/envs/protenix-baseline/bin/python
checkpoint=$root/yes-star24-home/projects/protenix-baseline/checkpoint/protenix-v2.pt
common=$root/yes-star24-home/projects/protenix-baseline/common
export PROTENIX_ROOT_DIR=$root/yes-star24-home/projects/protenix-baseline
export PYTHONPATH=$work:$repo/integrations/validation/cross_device_rng:$training/scripts:$repo
unset PROTENIX_TRIATTENTION_FLAGGEMS_PUBLIC PROTENIX_TRIATTENTION_FLAGGEMS
export CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=20260825
export PROTENIX_CROSS_DEVICE_RNG=1 PROTENIX_RNG_AUDIT=1 PROTENIX_RNG_SEED=20260825 PROTENIX_PATCH_GEOMETRY_RNG=1
export PROTENIX_DISABLE_PERMUTATION_ERROR_DUMP=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROTENIX_TF32_RESTORE_MODULES="" FLAG_GEMS_TRIANGLE_ATTENTION_FP16_TRAINING_SCRATCH=0
cd "$repo"
failures=0
for crop in 131 245; do
 for mode in fp32 tf32; do
  out=$work/training/$crop/$mode
  mkdir -p "$out"
  export PROTENIX_RNG_TRACE=$out/rng_trace.jsonl
  precision=fp32; candidate=public
  if [[ $mode != fp32 ]]; then precision=tf32; fi
  extra=()
  if [[ $mode == combined_tf32 ]]; then candidate=combined_64m; extra+=(--snapshot-reference "$work/training/$crop/tf32/first_step_local_snapshot.pt"); fi
  export PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION=ieee
  if [[ $precision == tf32 ]]; then export PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION=tf32; fi
  set +e
  timeout 3600 "$python" "$work/train_entry.py" --existing-tool "$training/scripts/training_parity_benchmark_20260825.py" \
   --precision "$precision" --candidate "$candidate" "${extra[@]}" --platform h100 --model protenix-v2 \
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
echo "$failures" > "$work/training.failures"
[[ $failures == 0 ]]
