# Experiment integrity audit

Date: 2026-11-06. **Overall verdict: WARN. Integrity status: warn.**

The frozen report's reported numbers and qualified conclusions are supported by the inspected artifacts. No remaining result-to-claim blocker was found. This is a **fresh same-family, provisional** review by GPT-6-Astra, reasoning `ultra`; it is not cross-family acceptance. The WARN concerns bounded experimental scope, aggregate-only verification, and the trace limitation below. It is not a universal training-accuracy PASS.

The reviewer read local source and aggregate artifacts, recomputed metrics and hashes, and reviewed pinned-parent diffs. No remote workload, environment installation, network exploration, raw-tensor transfer, or implementation/test modification was performed. Only the two audit reports and owned trace directory were written. One reused same-family explorer supplied source-only findings; the primary reviewer made the integrity judgments.

## Evidence identities

Abbreviations in file:line evidence: `P=/home/lgx1429541889/Protenix/comparison/upstream_merge_20261106`; `R=/home/lgx1429541889/upstream-merge-20261106/FlagGems`; `B=/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019`; `M=/home/lgx1429541889/Protenix/Protenix-main`. JSON result references also identify the relevant key where useful.

| Item | SHA256 / identity |
|---|---|
| Audited REPORT.md | `7a37a57d6742d1c5b3e35a9ef5fd4dc448b8106fe5b14705ec5d264afacda711` |
| Audited SUMMARY.json | `3c54dd491e8f2145724d8be36baf699887fe38f02b72fc89c0fba16a2876c9a6` |
| Tested source tree, 4,075 files | `5363b777bc6061d6492e07ec4d57afc56c9bd93e38944022d38d262720ffce7c` |
| Publication source tree, 4,075 files | `7a1d754e4955f60deea91c2282f4028d53390d24a5409dd80e73c50225918b10` |
| Pinned ours | `9583da7ab090e0f54498a23b2fe2e9a9e35abe7d` |
| Pinned upstream | `45339c4ad01d237d4fe66db197afb817eb9c6c60` |
| Common ancestor | `b550ddb996939ee1a30ba4a4b8f8b2d0556703a3` |

All 4,075 publication file hashes match the local repository. Tested/publication trees differ only in the preserved `_thead/__init__.py` comment; their executable ASTs are identical. The selected suite's `test_square.py` differs from publication only in blank-line formatting, also verified against preserved tested bytes. These are disclosed differences, not an assertion of byte identity. Evidence: `P/publication_source_manifest.json:1`, `P/policy_comment_provenance.json:1`, `P/test_formatting_provenance.json:1`, `P/REPORT.md:77`. Full audited-input hashes are in `EXPERIMENT_AUDIT.json`; they pin evidence, not an assertion that every upstream operator was semantically audited.

## A. Ground-truth provenance — PASS, for numerical-fidelity claims

The source comparisons generate seeded inputs and compare against independent CPU/native operations. The final registered precision probe implements CPU TF32 round-to-nearest-even with the bit expression `(bits + 4095 + ((bits >> 13) & 1)) & -8192`, then computes products and gradient references in double precision. It checks forward, all input gradients, and no-grad forward through actual torch entry points. Evidence: `P/compare_conflicts.py:29`, `P/compare_conflicts.py:83-105`, `P/check_precision_routes_v3.py:35-39`, `P/check_precision_routes_v3.py:71-102`.

H100 values are existing native-execution numerical references, not structural biological ground truth and not a fresh H100 run. Both recorded H100 snapshot hashes match the pinned baseline manifest. Training code loads experimental labels and checks assets/batches; raw remote labels and multi-GB snapshots were outside the authorized local review. The report now explicitly describes numerical consistency. Evidence: `P/analyze_upstream_merge_training_restored.py:29-47`, `B/current_bmm_policy_existing_H100_snapshot_manifest.json:1`, `B/../training_experimental.py:260-270`, `P/REPORT.md:34-38`.

## B. Score normalization — PASS

The full-vector metric is `sqrt(sum((candidate-reference)^2) / max(sum(reference^2), 1e-300))`. Raw difference/reference/actual squared norms, element counts, maximum errors and unequal-element counts are retained. No accuracy claim divides by the candidate's own maximum or mean. Module `share_of_full_squared_difference` is explicitly error attribution, not accuracy, and overlapping parent/child module buckets must not be summed as disjoint populations. Evidence: `B/analyze_linear_forward_training_20261023.py:24-60`.

All **448** relative-L2 entries in the final FP32/TF32 aggregate files recompute from their stored raw sums. The report's percentages multiply raw relative L2 by 100; its TF32 reductions recompute to 34.0752606535162% and 29.20307841804416%, correctly rounded to 34.08% and 29.20%. Losses and six comparison ratios per precision in SUMMARY match the source JSON exactly. Evidence: `P/build_report.py:17-18`, `P/build_report.py:54-78`, `P/build_report.py:107-108`, `P/REPORT.md:38-51`.

## C. Result existence and exact values — PASS

| Verified result | Observed scope/result |
|---|---|
| Fixed source comparison | 50 cases; ours 46 PASS, upstream 48 PASS, intermediate merged 50 PASS; 150 CSV rows match raw JSON |
| Final selected suite | 21 files; 186 collected = 161 passed + 25 skipped; 0 failures/errors; all 21 return codes zero |
| Final targeted regression | 26 actual JUnit cases = 13 Linear + 7 cdist + 6 MM; no skips/failures/errors |
| Registered autograd probe | 22/22 PASS with independent nonconstant upstream gradients |
| Registered precision probe | 10/10 PASS; same inputs used for FP32 then TF32; all forward/gradient relative-L2 checks below 2e-5 |
| Full model | Two completed optimizer steps for each precision; full step1 vectors and step0 Adam comparisons exist |

Evidence: `P/merged_source_comparison.json:1`, `P/conflict_source_cases.csv:1`, `P/validation_restored_results.json:1`, `P/targeted_regression_results.json:1`, `P/autograd_registration_restored.json:1`, `P/precision_routes_restored.json:1`, `P/REPORT.md:71-74`. The 50-case source result is an intermediate diagnostic, not final-package dispatch acceptance.

Exact full-vector errors below are **raw relative-L2 fractions**, not percentages:

| Metric against cached H100 | Old PPU FP32 | Final PPU FP32 | Old PPU TF32 | Final PPU TF32 |
|---|---:|---:|---:|---:|
| Step1 gradients | 1.5403599633673157e-05 | 1.9662595487683797e-05 | 0.012897880625567985 | 0.008502894183626328 |
| Step1 actual updates | 0.000429411421633987 | 0.0005626107896979529 | 0.06433917284225955 | 0.04555015374361353 |

Evidence: `P/training_comparison_fp32.json:443` and `:1636`; `P/training_comparison_tf32.json:443` and `:1636` (`metrics.gradients/updates.*.total.relative_l2`). FP32 residuals are small but larger than the old PPU values. TF32 gradient/update residuals are smaller on this trace, while its step1 absolute loss gap increases: final 4.580809593200684, old PPU 4.578239440917969, H100 4.5790205001831055. The report discloses both directions and does not assert a global numerical-accuracy PASS (`P/REPORT.md:42-51`; `P/SUMMARY.json`, `numerical_acceptance`).

### Timing and inherited gains

All seven-sample medians were recomputed; all 20 rows of `registered_api_timings.csv` match their JSON, including milliseconds-to-microseconds conversion. Final no-grad timings cover complete warmed forward calls, including allocation/needed layout work, but exclude context registration, compilation and backward. Source variants alternate across seven rounds; old-deployed and final-package timing records are separate sessions, not a paired significance study. Evidence: `P/compare_conflicts.py:107-130`, `P/check_precision_routes_v3.py:92-102`, `P/build_report.py:84-102`, `P/REPORT.md:57-67`.

The reported overlapping Unfold example matches the three-version diagnostic: 0.38461331526438397 / 3.6550798416137695 / 0.38311998049418133 ms (ours/upstream/merged). It preserves an existing optimization rather than inventing a new merge gain. Most final small-shape matrix interface medians are slower than the existing deployment; the table reports them without a blanket speedup. H100 same-input operator timings are absent and correctly shown as missing. Training has `timing_valid=false`; its audit/snapshot-heavy durations do not support an end-to-end speed claim (`P/REPORT.md:20,61-67,79`). Legacy `*_tf32` source rows use a different flag protocol and are explicitly excluded from final RNE performance evidence (`P/compare_conflicts.py:28,254-264`; `P/REPORT.md:22`).

## D. Executed code, dispatch and omitted stages — PASS with disclosed limitations

Training metrics are called for all gradient/update comparisons and both step0 Adam moments; their outputs are present (`P/analyze_upstream_merge_training_restored.py:55-67`). The observer counts then calls `func(*args, **kwargs)` unchanged and its recorded hash matches the inspected helper (`P/upstream_training_observer_restored_20261106.py:18-24`). It does not replace gradients or kernels. It only observes some ATen entries with direct CUDA tensor arguments, not every kernel or every nested/container argument. In particular, an AutogradCUDA override can execute above this observer; absence of a cdist counter is not proof that cdist did not run.

The isolated comparison loader (`P/compare_conflicts.py:20-41`) is distinguished from actual registration tests. Final F.linear records addmm once and mm twice for the tested forward/backward, with transpose/sum/view and no whole-layer Linear entry. Exported `flag_gems.linear` source hashes alone do not identify that executed path. Default `_thead` policy excludes `linear`; explicit include lists bypass vendor exclusions and direct calls remain possible. Thus explicit whole-layer Linear remains outside final default-training acceptance. Evidence: `R/src/flag_gems/runtime/backend/_thead/__init__.py:75-77`, `R/src/flag_gems/runtime/op_registrar.py:50-64`, `P/precision_routes_restored.json`, Linear `actual_ATen_calls`; `P/REPORT.md:17,67`.

The cdist regression was real: both self/cross registered VJPs failed in `autograd_registration_v2_before.json`, while the deployed source had an autograd registration absent from pinned git. The final backend retains all five deployed kernel ASTs, adds a context-scoped CUDA autograd Function, reduces both broadcast gradients, and guards empties. Final self/cross VJP relative-L2 errors are 7.204255966249223e-8 and 1.5405695661762224e-7. Source and registered tests support restoration of the missing path; the raw whole-model comparisons document the remaining differences. Evidence: `P/current_production_cdist_source.json:1`, `P/cdist_restoration_provenance.json:1`, `R/src/flag_gems/__init__.py:908-913`, `R/src/flag_gems/runtime/backend/_thead/ops/cdist.py:241-339`, `R/tests/test_cdist.py:43-79`, `P/autograd_registration_restored.json:1`.

The final public MM empty guard precedes the retained FP32 provider in `_thead/ops/mm.py:2492-2518`; it is not a new generic-MM guard. Six registered zero-M/K/N tests include backward and correctly shaped strided out buffers under both precision settings (`R/tests/test_mm.py:250-278`). Generic Linear's tuning key includes `ALLOW_TF32` (`R/src/flag_gems/ops/linear.py:116`). Compiled dispatch includes constexpr values (`R/src/flag_gems/utils/libentry.py:1605-1647`). The retained packed-RHS cache is queried only in enabled TF32-RNE mode; FP32 bypasses it, and its key includes storage/version/stride/device/stream (`R/src/flag_gems/runtime/backend/_thead/ops/mm_fp32.py:545-549,598-641,655-681`). This is source evidence plus tested precision switching, not exhaustive cache mutation/concurrency validation.

Failed/intermediate stages remain auditable:

- Initial source failures, repeated-output instability and upstream FP64 unfold failure remain in conflict records; they are not silently turned into passes.
- Upstream PPU MM compile timeout has no completed performance verdict (`P/override_compile_timeout.json`).
- Initial div/Square registration and CPU-reference fixture failures, later selected-suite failures, and upload-before-execution failure are retained (`validation_initial_failures.json`, `validation_v2_status.json`, `validation_selected_upload_failure.json`). The old suite runner could exit zero despite failed per-file results; the final runner asserts complete row count and every return code (`P/run_validation_suite_restored.py:53`).
- First model attempts failed on noncontiguous Linear layout before any optimizer step (`P/training_failed_layout.json`). The whole-layer and native-Linear controls later completed but both showed the cdist-related large gradient discrepancy; their files remain rejected adoption evidence (`training_whole_linear_*.json`, `training_native_linear_*.json`). `stopped_rejected_fp32_trial.json` says ALREADY_EXITED, not a successful cancellation.
- The first all-ones VJP smoke produced misleading near-zero-reference relative errors for Softmax/LayerNorm. The final probe uses fixed random upstream gradients and unchanged explicit absolute-plus-relative checks (`P/check_autograd_registration_final.py:37-58`).
- 3,994 Python AST parses and 25 export checks passed; pre_commit was unavailable and diff --check returned 2. These are not silently reported as passed format checks (`P/final_static_validation.json`; `P/REPORT.md:75`).

## E. Test, seed and training scope — WARN

The final analyzer requires identical actual batches and asset hashes across candidate, old PPU and H100, identical 3,338-event semantic RNG digests, and **before-step1 LR 1.8e-5 for all three sources**, not only the candidate (`P/analyze_upstream_merge_training_restored.py:29-37`). It verifies 4,174 initial parameter hashes, 4,170 gradient tensors and 4,174 updates, and the final total covers 464,441,841 gradient elements (`:44-61`). All observed provider hashes and the tested exclusion-policy hash match the immutable tested manifest. Imported loss/utils/permutation hashes match the local source mirror and baseline platform manifest.

There is one seed (20260825), two actual training inputs and two optimizer steps per precision. Step0 has LR=0 while Adam state still changes; step1 is the nonzero-LR update. Full-vector gradient/update comparison covers step1. Step0 Adam moments are compared with old PPU, not H100 (`:62-65`). The 17-row training pool is not 17 evaluated examples. The report does not claim convergence, multi-seed reliability or structure-prediction accuracy.

The 21-file suite uses `--quick --ref cpu --first-parameter-only`; it is not the complete upstream/local parameter-grid union. The 26 additional tests deliberately cover specified cases. cdist's new tests cover small FP32 p=2 forced non-MM distances, not every p, dtype, shape or native CUDA restoration-after-exit path; they do check repeated contexts and CPU-double gradients. Empty-MM tests use correctly shaped out buffers, not resize cases. The precision probe uses one shape per operator/precision. Evidence: `P/run_validation_suite_restored.py:19-36`, `R/tests/conftest.py:364-428`, `R/tests/test_cdist.py:43-79`, `R/tests/test_mm.py:250-278`, `P/check_precision_routes_v3.py:47-78`.

## F. Evaluation classification — PASS

| Evaluation | Classification | What it establishes |
|---|---|---|
| Seeded operator CPU/native/RNE comparisons | `synthetic_proxy`, numerical-reference subtype | Numerical contract on tested inputs |
| PPU/H100 gradient, update and loss comparison | `synthetic_proxy`, cross-device numerical-reference subtype | Fidelity to the cached native execution on this trace |
| Actual training objective and its experimental labels | `real_gt`, training-loss provenance only | Real-data training execution as documented by loader/asset audits; not a held-out biological benchmark |

No simulated physical evaluation or human evaluation is present. Native numerical outputs must not be relabeled as structural biological ground truth. The frozen report uses the qualified numerical framing (`P/REPORT.md:34-38`).

## Claim disposition and concrete follow-up

- **Supported within scope:** conflict resolution/provenance; listed final selected/targeted/registered tests; default Linear decomposition; repaired cdist registered gradients; exact two-step aggregate comparisons; disclosed source-call timings.
- **Qualified only:** smaller TF32 gradient/update errors on this one trace. FP32 gradient/update errors and TF32 step1 loss gap increase; preserve that disclosure in every derived summary.
- **Unsupported by this evidence:** universal numerical non-regression, full-suite acceptance, long-run convergence, general speedup, new H100 validation, structural biological accuracy, or adoption of explicitly enabled whole-layer Linear.
- Future broader claims require the corresponding additional shapes/dtypes/seeds or clean performance runs. No extra experiment is required to retain the present bounded report. Complete format hooks remain a separate unexecuted check.

## Forensic trace and assurance limit

Trace: `.aris/traces/experiment-audit/2026-11-06_run01/`. The primary request, final authored response, hashes and deterministic checks are preserved. Auxiliary source-review calls 002, 003 and 008 have request/response records. For call 006, the exact task requests and a fresh complete response are preserved, but an earlier clone-response wording was compacted before archival; the restatement is explicitly labeled and is not claimed as recovered verbatim. This historical auxiliary trace gap is disclosed rather than silently omitted.

The reviewer did not independently recompute vector statistics from the original remote tensors. It verified the implemented metric, actual artifact existence, stored raw-sum arithmetic, result/claim agreement and source/protocol provenance. These limits, and same-family review, make assurance provisional. Audit generated at 2026-11-06T06:50:43.471735+00:00.
