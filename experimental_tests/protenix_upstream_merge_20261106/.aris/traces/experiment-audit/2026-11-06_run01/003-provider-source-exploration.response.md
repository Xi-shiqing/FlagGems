Current-source findings, read-only:

- **Backward layout fix is appropriate:** `_thead/ops/linear_backward.py:165-166` now materializes contiguous input/gradient before reshaping. Both new cases in `tests/test_linear.py:151-182` are supported: batch-transposed tensors and stride-2 output gradients. Tests use registered `aten.linear_backward`, CPU float64 references, and TF32 disabled.
- **Original failure is preserved:** `training_failed_layout.json` records both precision runs failing at the old `grad_output.view(...).contiguous()` with zero optimizer steps. These runs provide no completed-training comparison.
- **Forward layout gaps remain:** `_thead/ops/linear.py:392` still calls `input.view(-1,K)` on strict-FP32 small-row inputs; valid batch-transposed `(3,5,128)` input cannot flatten this way. Its newly added TF32 branch at `:370` uses `reshape(-1,0)` for zero-width inputs, which is ambiguous. Generic `ops/linear.py:492` has the same empty/noncontiguous limitation. The strict-path view issue is inherited upstream; the TF32 branch is merge-added. No local Torch was available for execution.
- **Potential empty-reduction MM hazard:** retained `mm_fp32.py` has no `K==0` guard before general dispatch (`:715-746`). `prev_multiple_of(0,32)` yields `-32` (`:34-37`), then loop-peeling reads negative reduction offsets with only `rk<K` masking (`:106-120`). Public `_thead/ops/mm.py:2493-2494` sends FP32 inputs directly here. Treat empty-K support as unverified and add a guard before claiming general semantic coverage.
- **No precision-cache collision found in inspected source:** retained MM/BMM/AddMM pass `ALLOW_TF32` as `tl.constexpr`; `libentry.py:1622-1647` includes keyword/default constexpr values in the cache key. Packed MM RHS cache is used only when TF32 RNE packing is enabled (`mm_fp32.py:545-548,658-668`); key includes storage identity, version, shape, stride, dtype, device, and stream (`:609-612`). It cannot feed packed values to the strict path through these branches.
- **Actual public routes:** `_thead/ops/__init__.py:57-58` exports backend linear/backward; `_FULL_CONFIG` registers both (`flag_gems/__init__.py:1445-1446`). FP32 backward delegates to public `flag_gems.mm` (`linear_backward.py:168-182`), which selects the retained provider. FP32 TF32 forward calls retained providers explicitly (`linear.py:365-374`).

Replacement-training evidence:

- `analyze_upstream_merge_training_layout.py` corrects the prior “complete two-step gradients” docstring. Its metrics remain step1 gradients/updates against cached H100 and old PPU, plus step0 Adam moments against old PPU.
- It still checks candidate LR only; old/H100 LR receipts remain necessary.
- `queue_training_layout.py` requires precision and layout validation PASS and creates a distinct output directory.
- `queue_training_analysis_layout.py:12-27` requires successful training exit/COMPLETE and performs CPU-only analysis.
- Current `linear_backward.py` hash matches `merged_layout_fix_all_source_hashes.json`: `f6c8bdc497c7e470b7acd80c81c571d29c3a34e4dc60d59e7577c72edead53ae`. Retained MM/BMM/AddMM hashes match previous providers exactly.

`validation_selected_results.json` now reports **150 passed, 24 skipped, zero failures/errors**, across 174 selected tests for **merged_release**. Its four linear cases predate the added backward-layout tests; require the new layout-validation receipt for those. Final FP32/TF32 comparison results and three-source LR receipts remain pending.
