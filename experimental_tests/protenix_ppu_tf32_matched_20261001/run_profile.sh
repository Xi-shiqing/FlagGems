#!/usr/bin/env bash
set -euo pipefail
root=/mnt/cpfs/xishiqing
exp=$root/ppu_tf32_rne_20260923
vendor=$exp/vendor_gemm_20260930
source "$vendor/env_ppu.sh"
prior=$exp/jittor_reference_20261001
work=$exp/matched_tf32_20261001
export PYTHONPATH=$prior/candidate_unsplit:$work:$vendor:$root/shared_rng:$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PROTENIX_CROSS_DEVICE_RNG=1 PROTENIX_RNG_AUDIT=1 PROTENIX_RNG_SEED=101
export PROTENIX_TRIATTENTION_FLAGGEMS_PUBLIC=1 PROTENIX_TRIATTENTION_FLAGGEMS=0
export PROTENIX_TRIATTENTION_PPU_STREAMING=0 PROTENIX_TRIATTENTION_PPU_SMALL=0
export MM_PROFILE=1 MM_DIAGNOSTIC_TRACE=1
export MM_SHARED_ESM_ROOT=$root/ppu_mm_crossdevice_20260920/shared_assets
cd "$repo"
for case_id in G02; do
  model=protenix_base_default_v1.0.0
  out=$work/profile/$case_id
  mkdir -p "$out"
  export PROTENIX_RNG_TRACE=$out/rng_trace.jsonl
  export PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION=tf32
  set +e
  timeout 2400 python3 "$vendor/legacy_msa_harness.py" \
    --entrypoint "$vendor/audit_gems_entrypoints.py" --numpy-root "$vendor/numpy126" \
    --work-root "$out/msa_worker" --gems-entrypoint "$work/pair_profile.py" \
    --entry-audit "$out/gems_entry_audit.json" \
    --harness "$exp/generic_pipeline_20260926/benchmark_model.py" \
    --baseline "$prior/candidate" --references "$vendor/full_gems/$case_id/$model" \
    --paired-output "$out/paired.json" --numeric-candidate \
    --repo-root "$repo" --input "$root/ppu_mm_generalization_20260919/inputs/$case_id/input.json" \
    --out-root "$out" --model "$model" --use-msa true --use-template false --use-rna-msa false \
    --dtype fp32 --seed 101 --cycle 1 --step 1 --sample 1 \
    --triatt-kernel triattention --trimul-kernel torch \
    --enable-cache true --enable-fusion true --enable-tf32 true \
    --warmup 2 --repeats 2 --use-flag-gems --save-predictions > "$out/run.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$out/exit_code"
  echo "MATCHED_PPU $case_id exit=$rc"
  [[ $rc == 0 ]]
done
