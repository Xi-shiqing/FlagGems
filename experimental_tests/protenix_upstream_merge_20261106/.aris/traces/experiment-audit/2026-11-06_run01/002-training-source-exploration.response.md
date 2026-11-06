Read-only source review complete. No final integrity verdict issued.

Use `P=/home/lgx1429541889/Protenix/comparison/upstream_merge_20261106`, `B=/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019`.

Actionable findings:

1. **Suite process exit is not a pass criterion.** `P/run_validation_suite_selected.py:44-48` records failed subprocess codes but never propagates failure. `P/validation_v2_status.json:2` reports outer exit `0` despite square and libentry failures. Require all 19 expected file records, per-file zero return codes, and JUnit outcomes.

2. **Training route PASS does not certify backend routes.** Runtime uses `--skip-route-audit` (`B/linear_x3_environment_20261023.sh:40`). `.../ppu_full_acceptance_20261005/train_route_entry.py:46` emits PASS when no permutation warning occurs, even with profiling disabled. `P/analyze_upstream_merge_training.py:37` checks this status only. Describe that check as warning monitoring; use independently verified dispatch/provider evidence for route claims.

3. **“Complete two-step gradients” overstates analyzed scope.** Analyzer docstring says this at `P/analyze_upstream_merge_training.py:1`; actual comparisons cover step1 gradients/updates (`:54-60`) and step0 Adam moments (`:61-64`). Step0 preclip gradients are saved (`B/train_softmax_forward_fidelity_20261023.py:66-79`) but not compared by the new analyzer. Rename scope or add the omitted comparison.

4. **LR equality is not fully asserted by the new analyzer.** Candidate step1 LR is checked against `[1.8e-5]` (`P/analyze_upstream_merge_training.py:36`), but old PPU/H100 LR receipts are not read. Verify their `actual_parameter_audit.json` and selected protocol fields before claiming identical learning rates/settings. Candidate step0 LR `[0.0]` is directly asserted (`:63`).

5. **Selected pytest scope is narrow.** `P/run_validation_suite_selected.py:30-34` uses `--quick --first-parameter-only`, except the square regression. `FlagGems/tests/conftest.py:364-428` selects one combination per test function and prefers bfloat16. This is a smoke suite, not full FP32/TF32 dtype/shape coverage.

6. **Training timings cannot support speed claims.** Inherited receipt explicitly records `timing_valid=False` (`B/train_softmax_forward_fidelity_20261023.py:110`). Recorded event/wall metrics are milliseconds around the wrapped training step (`.../training_experimental.py:428-441`), including audit copies/hooks; launcher timing uses seconds (`:552,647`). Checkpoint/evaluation time is excluded (`:616`).

Verified source properties:

- Observer does not replace gradients or kernels: `P/upstream_training_observer_20261106.py:18-24` records calls then directly invokes `func`. Counts cover only CUDA tensors directly present in positional `args`; nested tensor containers and keyword-only tensors can be absent from the counter (`:20`). Do not call this an exhaustive ATen trace.
- Metric normalization uses the reference vector norm, with raw squared differences, norms, max absolute error, and differing-element counts (`B/analyze_linear_forward_training_20261023.py:24-60`). No self-normalized accuracy score found. Imported `rng`, `finish`, `metrics`, and `moments` functions are exercised.
- Analyzer asserts matching batches, asset hashes, ordered RNG count `3338`, 4174 initial parameter hashes, 4170 gradient tensors, 4174 update tensors, and `464441841` gradient elements (`P/analyze_upstream_merge_training.py:30-45,60`).
- H100 snapshot hashes are pinned in `B/current_bmm_policy_existing_H100_snapshot_manifest.json:2-12`; analyzer verifies the applicable hash (`P/analyze_upstream_merge_training.py:41-43`). These are cached native numerical references, **not structural biological ground truth**.
- Training data loader checks a disjoint experimental-label manifest and each asset hash (`.../training_experimental.py:260-270`), hashes actual batch contents (`:309-323`), and records unmodified native loss outputs (`:329-348`). Raw fixtures/manifest were not locally available for independent label inspection.
- Planned training scope is one seed (`20260825`), one run each FP32/TF32, two steps, Protenix-v2, crop 693, 17-row training pool, 48 Pairformer blocks, four MSA blocks (`B/linear_x3_environment_20261023.sh:26,40-50`). Two steps do not mean all 17 examples were evaluated.

Pending evidence:

- Current local progress snapshots show training active without final reports/comparisons and selected suite only `copy`/`div` completed (`P/training_and_suite_progress.json`).
- Need final candidate training receipts, comparison JSONs, old/H100 LR/protocol receipts, final selected per-file/JUnit results, and source-hash correspondence before endorsing completed training or precise numerical claims.
- Preserve recorded prior failures: initial div autograd failure, CPU-reference fixture issues, v2 square/libentry failures, and selected-suite upload failure. Later successful stages must be linked to their exact package and test versions.
