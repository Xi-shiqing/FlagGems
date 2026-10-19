#!/usr/bin/env bash
# Public launcher: path configuration only; measured kernel sources are unchanged.
set -euo pipefail
: "${PTX_MUSA_HOME:?Set the existing Protenix MUSA deployment directory}"
: "${PTX_GPU:?Select a free container-visible MUSA GPU}"
: "${INPUT_NAME:?Set the prepared input name, for example 7b8b}"
task_root="$(cd "$(dirname "$0")" && pwd)"
source "$PTX_MUSA_HOME/env.sh"
export EXPERIMENT_ROOT="$task_root"
export EXPERIMENT_MODE="${EXPERIMENT_MODE:-recheck_native}"
export BENCH_REPEATS="${BENCH_REPEATS:-6}"
export BENCH_SEED="${BENCH_SEED:-101}"
export MUSA_AMP_ALIGNMENT="${MUSA_AMP_ALIGNMENT:-1}"
export MUSA_TF32="${MUSA_TF32:-true}"
export PRECISION_AUDIT="${PRECISION_AUDIT:-1}"
export FROZEN_MODE="${FROZEN_MODE:-read}"
export MUSA_SYNC_BOUNDARY=cycle ATTENTION_COMPAT=none SAFE_TRI_MUL=0
cd "$PTX_MUSA_HOME/src"
exec python "${BENCHMARK_ENTRY:-$EXPERIMENT_ROOT/benchmark.py}" \
  --input_json_path "$EXPERIMENT_ROOT/inputs/$INPUT_NAME.json" \
  --dump_dir "$EXPERIMENT_ROOT/outputs/$EXPERIMENT_MODE/gpu_$PTX_GPU" \
  --model_name protenix_base_default_v1.0.0 --seeds "$BENCH_SEED" --dtype bf16 \
  --use_msa true --use_template false --use_rna_msa false \
  --triangle_attention torch --triangle_multiplicative torch \
  --enable_tf32 "$MUSA_TF32" --enable_diffusion_shared_vars_cache true --enable_efficient_fusion true \
  --model.N_cycle 10 --sample_diffusion.N_step 200 --sample_diffusion.N_sample 5 "$@"
