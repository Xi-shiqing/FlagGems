"""Write only the auditor-owned reports and forensic trace."""
from pathlib import Path
import ast
import csv
import datetime
import hashlib
import json
import math
import statistics

P = Path('/home/lgx1429541889/Protenix/comparison/upstream_merge_20261106')
R = Path('/home/lgx1429541889/upstream-merge-20261106/FlagGems')
B = Path('/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019')
M = Path('/home/lgx1429541889/Protenix/Protenix-main')
T = P/'.aris/traces/experiment-audit/2026-11-06_run01'
read = lambda n: json.loads((P/n).read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
expected_report = '7a37a57d6742d1c5b3e35a9ef5fd4dc448b8106fe5b14705ec5d264afacda711'
expected_summary = '3c54dd491e8f2145724d8be36baf699887fe38f02b72fc89c0fba16a2876c9a6'
assert sha(P/'REPORT.md') == expected_report
assert sha(P/'SUMMARY.json') == expected_summary

def line(path, needle):
    return next(i for i, text in enumerate(path.read_text().splitlines(), 1) if needle in text)

training = {m: read('training_comparison_'+m+'.json') for m in ('fp32', 'tf32')}
suite = read('validation_restored_results.json')
target = read('targeted_regression_results.json')
precision = read('precision_routes_restored.json')
autograd = read('autograd_registration_restored.json')
summary = read('SUMMARY.json')
tested = read('merged_restored_all_source_hashes.json')
publication = read('publication_source_manifest.json')
assert len(tested) == len(publication['files']) == 4075
assert hashlib.sha256(json.dumps(tested, sort_keys=True).encode()).hexdigest() == publication['tested_tree_sha256']
assert hashlib.sha256(json.dumps(publication['files'], sort_keys=True).encode()).hexdigest() == publication['tree_sha256']
assert all(sha(R/'src'/name) == value for name, value in publication['files'].items())
delta = [n for n in tested if tested[n] != publication['files'][n]]
assert delta == ['flag_gems/runtime/backend/_thead/__init__.py']
old = (P/'tested_source_bytes/thead_init.py.txt').read_text()
new = (R/'src'/delta[0]).read_text()
assert ast.dump(ast.parse(old), include_attributes=False) == ast.dump(ast.parse(new), include_attributes=False)
assert hashlib.sha256(old.encode()).hexdigest() == tested[delta[0]]
sq_old = (P/'tested_source_bytes/test_square.py.txt').read_text()
sq_new = (R/'tests/test_square.py').read_text()
assert ast.dump(ast.parse(sq_old), include_attributes=False) == ast.dump(ast.parse(sq_new), include_attributes=False)
assert hashlib.sha256(sq_old.encode()).hexdigest() == suite['source_hashes']['tests/test_square.py']
for name, value in suite['source_hashes'].items():
    if name != 'tests/test_square.py':
        assert sha(R/name) == value

counts = {k: sum(int(group[k]) for groups in suite['junit'].values() for group in groups)
          for k in ('tests', 'skipped', 'errors', 'failures')}
counts['passed'] = counts['tests']-counts['skipped']-counts['errors']-counts['failures']
assert counts == suite['counts'] == summary['selected_suite']
assert counts == dict(tests=186, skipped=25, errors=0, failures=0, passed=161)
assert len(suite['per_file']) == len(suite['junit']) == 21
assert all(x['returncode'] == 0 for x in suite['per_file'])
assert len(target['test_cases']) == 26
assert sum(int(x['tests']) for x in target['testsuite_attributes']) == 26
assert all(int(x[k]) == 0 for x in target['testsuite_attributes'] for k in ('errors', 'failures', 'skipped'))
assert len(autograd['rows']) == 22 and all(x['status'] == 'PASS' for x in autograd['rows'])
assert len(precision['cases']) == 10 and all(x['status'] == 'PASS' for x in precision['cases'])
assert all(len(x['device_ms']) == 7 and statistics.median(x['device_ms']) == x['median_ms'] for x in precision['cases'])

ratios = [0]
def check_metrics(obj):
    if isinstance(obj, dict):
        if {'difference_squared', 'reference_squared', 'relative_l2'} <= set(obj):
            expected = math.sqrt(obj['difference_squared']/max(obj['reference_squared'], 1e-300))
            assert math.isclose(expected, obj['relative_l2'], rel_tol=1e-14, abs_tol=0)
            ratios[0] += 1
        for value in obj.values():
            check_metrics(value)
    elif isinstance(obj, list):
        for value in obj:
            check_metrics(value)

for mode, data in training.items():
    check_metrics(data)
    assert data['status'] == 'COMPLETE_FULL_TRAINING_COMPARISON'
    assert data['actual_training_steps'] == 2 and data['actual_learning_rate_step1'] == 1.8e-5
    assert data['uniform_RNG']['count'] == 3338
    assert data['execution']['optimizer_steps'] == 2 and data['execution']['timing_valid'] is False
    assert data['new_H100_execution'] is False
    observation = data['execution']['registration_observation']
    assert observation['gradient_or_kernel_replacement_by_observer'] is False
    assert observation['PPU_exclusion_policy_sha256'] == tested['flag_gems/runtime/backend/_thead/__init__.py']
    assert data['execution']['helper_sha256'] == sha(P/'upstream_training_observer_restored_20261106.py')
    for value in {**observation['public_sources'], **observation['retained_current_providers']}.values():
        relative = 'flag_gems/'+value['path'].split('/flag_gems/', 1)[1]
        assert value['sha256'] == tested[relative]
    for field in ('new_losses', 'old_PPU_losses', 'H100_losses'):
        assert data[field] == summary['training'][mode]['losses'][field]
    for kind, prefix in [('gradients', 'gradient'), ('updates', 'update')]:
        for comparison, metrics in data['metrics'][kind].items():
            assert summary['training'][mode][prefix+'_relative_percent'][comparison] == metrics['total']['relative_l2']*100
assert ratios[0] == 448

key_names = '''PLAN.md REPORT.md SUMMARY.json REPORT_DRAFT.md build_report.py
git_inventory.json merge_conflicts.json merge_attempt.txt integration_triage.json
compare_conflicts.py compare_merged.py compare_linear_v2.py compare_overrides.py
check_index_override.py check_precision_routes.py check_precision_routes_v2.py check_precision_routes_v3.py check_registered_gradients.py
conflict_comparison.json merged_source_comparison.json linear_precision_comparison.json index_override_comparison.json
precision_routes_existing_control.json precision_routes_selected.json precision_routes_ready.json precision_routes_restored.json
registration_validation_progress.json override_compile_timeout.json validation_initial_failures.json validation_v2_status.json validation_selected_upload_failure.json
retained_training_operator_provenance.json deployed_operator_inventory.json deployed_full_source_manifest.json current_production_cdist_source.json cdist_restoration_provenance.json
merged_release_all_source_hashes.json merged_release_package_manifest.json merged_release_deployment.json
merged_layout_fix_all_source_hashes.json merged_layout_fix_package_manifest.json merged_layout_fix_deployment.json
merged_complete_all_source_hashes.json merged_complete_package_manifest.json merged_complete_deployment.json
merged_ready_all_source_hashes.json merged_ready_package_manifest.json merged_ready_deployment.json
merged_adopted_all_source_hashes.json merged_adopted_package_manifest.json merged_adopted_deployment.json
merged_restored_all_source_hashes.json merged_restored_package_manifest.json merged_restored_deployment.json publication_source_manifest.json
prepare_restored_package.py stage_packages.py prepare_release_package.py
train_upstream_merge_20261106.py upstream_training_observer_20261106.py queue_training_20261106.py analyze_upstream_merge_training.py
analyze_upstream_merge_training_layout.py queue_training_layout.py train_upstream_merge_layout.py queue_training_analysis_layout.py
train_upstream_merge_complete.py queue_training_complete.py analyze_upstream_merge_training_complete.py
queue_training_ready.py train_upstream_merge_ready.py analyze_upstream_merge_training_ready.py
analyze_upstream_merge_training_native_linear.py queue_training_native_linear.py queue_training_analysis_native_linear.py
analyze_upstream_merge_training_restored.py train_upstream_merge_restored.py queue_training_restored.py queue_training_analysis_restored.py upstream_training_observer_restored_20261106.py
run_validation_suite_selected.py run_validation_suite_ready.py run_validation_suite_restored.py queue_restored_validation.py queue_autograd_registration_restored.py
check_autograd_registration.py check_autograd_registration_v2.py check_autograd_registration_final.py
validation_selected_results.json validation_ready_results.json validation_restored_results.json targeted_regression_results.json
ready_validation_verdict.json restored_validation.json restored_validation_progress.json final_static_validation.json
training_failed_layout.json training_initial_failures_and_suite.json training_adoption_progress.json
training_native_linear_fp32.json training_native_linear_tf32.json training_whole_linear_fp32.json training_whole_linear_tf32.json
training_comparison_fp32.json training_comparison_tf32.json training_restored_latest.json
autograd_registration_before.json autograd_registration_v2_before.json autograd_registration_restored.json stopped_rejected_fp32_trial.json
test_formatting_provenance.json policy_comment_provenance.json
conflict_source_cases.csv registered_api_timings.csv training_comparison.csv
restored_delta.tar.gz restored_validation_tests.tar.gz'''.split()
inputs = {str(P/name): 'sha256:'+sha(P/name) for name in key_names if (P/name).is_file()}
for file in sorted((P/'variants').rglob('*.py')):
    inputs[str(file)] = 'sha256:'+sha(file)
for relative in ['analyze_linear_forward_training_20261023.py', 'analyze_softmax_forward_fidelity_training_20261023.py',
                 'train_softmax_forward_fidelity_20261023.py', 'linear_x3_environment_20261023.sh',
                 'current_bmm_policy_existing_H100_snapshot_manifest.json', 'matching_candidate_manifests.json']:
    inputs[str(B/relative)] = 'sha256:'+sha(B/relative)
inputs[str(B.parent/'training_experimental.py')] = 'sha256:'+sha(B.parent/'training_experimental.py')
for relative in ['protenix/model/loss.py', 'protenix/utils/torch_utils.py', 'protenix/utils/permutation/permutation.py']:
    inputs[str(M/relative)] = 'sha256:'+sha(M/relative)
for relative in ['src/flag_gems/__init__.py', 'src/flag_gems/runtime/op_registrar.py', 'src/flag_gems/utils/libentry.py',
                 'src/flag_gems/runtime/backend/_thead/__init__.py', 'src/flag_gems/runtime/backend/_thead/ops/__init__.py',
                 'src/flag_gems/runtime/backend/_thead/ops/cdist.py', 'src/flag_gems/runtime/backend/_thead/ops/mm.py',
                 'src/flag_gems/runtime/backend/_thead/ops/mm_fp32.py', 'src/flag_gems/runtime/backend/_thead/ops/bmm_fp32.py',
                 'src/flag_gems/runtime/backend/_thead/ops/addmm_fp32.py', 'src/flag_gems/runtime/backend/_thead/ops/softmax_fp32.py',
                 'src/flag_gems/runtime/backend/_thead/ops/linear.py', 'src/flag_gems/runtime/backend/_thead/ops/linear_backward.py',
                 'src/flag_gems/ops/linear.py', 'src/flag_gems/ops/pow.py', 'src/flag_gems/ops/cdist.py',
                 'tests/test_linear.py', 'tests/test_cdist.py', 'tests/test_mm.py', 'tests/test_square.py', 'tests/conftest.py']:
    inputs[str(R/relative)] = 'sha256:'+sha(R/relative)

grad = lambda mode, comparison: training[mode]['metrics']['gradients'][comparison]['total']['relative_l2']
upd = lambda mode, comparison: training[mode]['metrics']['updates'][comparison]['total']['relative_l2']
metric_lines = {mode: {kind: line(P/f'training_comparison_{mode}.json', '"relative_l2": '+str(training[mode]['metrics'][kind]['new_vs_H100']['total']['relative_l2']))
                      for kind in ('gradients', 'updates')} for mode in training}

checks = {
 'gt_provenance': {'status':'PASS', 'details':'Independent native CPU/CPU-RNE references and cached H100 numerical outputs are correctly treated as numerical-fidelity proxies. Training loads experimental-data labels; this audit did not inspect raw remote labels or tensors.', 'evidence':['P/check_precision_routes_v3.py:35-39','P/check_precision_routes_v3.py:71-78','P/analyze_upstream_merge_training_restored.py:29-47','P/REPORT.md:34-38','B/../training_experimental.py:260-270']},
 'score_normalization': {'status':'PASS','details':'Relative L2 divides by the reference norm, preserves raw squared differences/norms, and is converted to percent by multiplying by 100. All 448 final training ratios recompute. Error-attribution shares are not accuracy scores.','evidence':['B/analyze_linear_forward_training_20261023.py:24-60','P/build_report.py:17-18','P/REPORT.md:38']},
 'result_existence': {'status':'PASS','details':'Final report and summary numbers match existing completed records; JUnit counts, all timing CSV rows, 150 conflict-source CSV rows, source tree digests and provider hashes were checked. Failed and abandoned candidates remain distinct.','evidence':['P/REPORT.md:20','P/REPORT.md:40-79','P/validation_restored_results.json:1','P/targeted_regression_results.json:1','P/training_comparison_fp32.json:1','P/training_comparison_tf32.json:1']},
 'dead_code': {'status':'PASS','details':'Training metric routines are actually called for gradients, updates and step0 Adam moments; emitted results exist. Registered torch paths are tested separately from isolated source tests. Observer counts then forwards the operation unchanged. Excluded whole-layer Linear is explicitly disclosed.','evidence':['P/analyze_upstream_merge_training_restored.py:55-67','P/upstream_training_observer_restored_20261106.py:18-24','P/check_precision_routes_v3.py:79-102','R/src/flag_gems/runtime/backend/_thead/__init__.py:75-77','P/REPORT.md:17']},
 'scope': {'status':'WARN','details':'One seed, two actual training inputs/steps per precision; full vectors are compared only at step1 and step0 Adam moments only against old PPU. Selected 21-file first-parameter-only tests, small precision probes and cached H100 do not establish full-suite coverage, convergence, broad speedup or biological accuracy.','evidence':['P/REPORT.md:36','P/REPORT.md:57-79','P/run_validation_suite_restored.py:19-36','P/analyze_upstream_merge_training_restored.py:44-65','R/tests/conftest.py:364-428']},
 'eval_type': {'status':'PASS','details':'Numerical operator and cross-device fidelity = synthetic_proxy (numerical-reference subtype); training objective uses real_gt labels as documented by source/asset audits. No biological structural-accuracy evaluation or human evaluation is claimed.','classification':['synthetic_proxy:numerical_reference','real_gt:training_loss_only'],'evidence':['P/REPORT.md:34-38','P/check_precision_routes_v3.py:35-78','P/analyze_upstream_merge_training_restored.py:42-58','M/protenix/model/loss.py:259-284']}
}

body = f'''# Experiment integrity audit

Date: 2026-11-06. **Overall verdict: WARN. Integrity status: warn.**

The frozen report's reported numbers and qualified conclusions are supported by the inspected artifacts. No remaining result-to-claim blocker was found. This is a **fresh same-family, provisional** review by GPT-6-Astra, reasoning `ultra`; it is not cross-family acceptance. The WARN concerns bounded experimental scope, aggregate-only verification, and the trace limitation below. It is not a universal training-accuracy PASS.

The reviewer read local source and aggregate artifacts, recomputed metrics and hashes, and reviewed pinned-parent diffs. No remote workload, environment installation, network exploration, raw-tensor transfer, or implementation/test modification was performed. Only the two audit reports and owned trace directory were written. One reused same-family explorer supplied source-only findings; the primary reviewer made the integrity judgments.

## Evidence identities

Abbreviations in file:line evidence: `P={P}`; `R={R}`; `B={B}`; `M={M}`. JSON result references also identify the relevant key where useful.

| Item | SHA256 / identity |
|---|---|
| Audited REPORT.md | `{expected_report}` |
| Audited SUMMARY.json | `{expected_summary}` |
| Tested source tree, 4,075 files | `{publication['tested_tree_sha256']}` |
| Publication source tree, 4,075 files | `{publication['tree_sha256']}` |
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
| Step1 gradients | {grad('fp32','old_vs_H100')} | {grad('fp32','new_vs_H100')} | {grad('tf32','old_vs_H100')} | {grad('tf32','new_vs_H100')} |
| Step1 actual updates | {upd('fp32','old_vs_H100')} | {upd('fp32','new_vs_H100')} | {upd('tf32','old_vs_H100')} | {upd('tf32','new_vs_H100')} |

Evidence: `P/training_comparison_fp32.json:{metric_lines['fp32']['gradients']}` and `:{metric_lines['fp32']['updates']}`; `P/training_comparison_tf32.json:{metric_lines['tf32']['gradients']}` and `:{metric_lines['tf32']['updates']}` (`metrics.gradients/updates.*.total.relative_l2`). FP32 residuals are small but larger than the old PPU values. TF32 gradient/update residuals are smaller on this trace, while its step1 absolute loss gap increases: final 4.580809593200684, old PPU 4.578239440917969, H100 4.5790205001831055. The report discloses both directions and does not assert a global numerical-accuracy PASS (`P/REPORT.md:42-51`; `P/SUMMARY.json`, `numerical_acceptance`).

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

The reviewer did not independently recompute vector statistics from the original remote tensors. It verified the implemented metric, actual artifact existence, stored raw-sum arithmetic, result/claim agreement and source/protocol provenance. These limits, and same-family review, make assurance provisional. Audit generated at {now}.
'''

findings = [
 {'id':'F1','severity':'high','state':'resolved_for_reported_default_scope','title':'Lost deployed cdist autograd','evidence':['autograd_registration_v2_before.json','current_production_cdist_source.json','autograd_registration_restored.json','training_comparison_fp32.json','training_comparison_tf32.json'],'correction':'Restored deployed kernels with scoped public autograd and reran registered VJPs plus whole-model comparisons.'},
 {'id':'F2','severity':'medium','state':'disclosed','title':'Numerical changes are mixed','correction':'Retain exact old/new errors; do not label completion a global numerical PASS. FP32 gradient/update errors and TF32 step1 loss gap increase.'},
 {'id':'F3','severity':'medium','state':'disclosed','title':'Default Linear differs from exported whole-layer implementation','correction':'Keep default decomposition and explicit-opt-in scope clear; actual route counters, not exported source name alone, describe the tested path.'},
 {'id':'F4','severity':'medium','state':'resolved','title':'Legacy TF32 diagnostics and inherited timings could be overclaimed','correction':'Report separates legacy flag-mode source diagnostics, current RNE registered measurements, inherited gains and invalid training timings.'},
 {'id':'F5','severity':'medium','state':'resolved','title':'Outer zero exit did not imply all old validation files passed','correction':'Final suite propagates aggregate failures and all 21 per-file/JUnit records were checked; old failures remain preserved.'},
 {'id':'F6','severity':'medium','state':'scope_limitation','title':'Bounded numerical pilot and selected tests','correction':'Restrict claims to one seed/two inputs/two steps and listed parameter selections; larger claims require broader evidence.'},
 {'id':'F7','severity':'low','state':'disclosed','title':'Formatting checks incomplete','correction':'pre_commit unavailable and diff --check=2 retained and disclosed; syntax/export checks passed separately.'},
 {'id':'F8','severity':'low','state':'trace_limitation','title':'Historical auxiliary response gap','correction':'Recovered requests and new response are archived; original compacted clone prose is not represented as verbatim.'},
]
claims = [
 {'id':'merge_and_source_provenance','impact':'supported'},
 {'id':'listed_operator_checks','impact':'supported'},
 {'id':'two_step_numerical_comparison','impact':'supported_with_scope'},
 {'id':'TF32_gradient_update_error_reduction','impact':'needs_single_trace_qualifier'},
 {'id':'global_accuracy_non_regression','impact':'unsupported_and_not_asserted'},
 {'id':'end_to_end_speedup','impact':'unsupported_and_not_asserted'},
 {'id':'biological_structural_accuracy','impact':'unsupported_and_not_asserted'},
 {'id':'explicit_whole_layer_linear_adoption','impact':'unsupported_and_not_asserted'},
]
result = {
 'audit_skill':'experiment-audit','verdict':'WARN','reason_code':'bounded_scope_and_provisional_assurance',
 'summary':'Frozen report numbers and qualified claims match inspected artifacts. Mixed numerical changes, limited test/training scope, aggregate-only verification and an auxiliary trace gap preclude unqualified acceptance.',
 'audited_input_hashes':inputs,'trace_path':'.aris/traces/experiment-audit/2026-11-06_run01/',
 'agent_id':'/root/audit_upstream_merge_20261106','verdict_id':'/root/audit_upstream_merge_20261106/2026-11-06_run01',
 'executor_model':'not_exposed_to_reviewer','executor_family':'openai','reviewer_model':'gpt-6-astra','reviewer_family':'openai','reviewer_reasoning':'ultra',
 'review_independence':'same-family','acceptance_status':'provisional','generated_at':now,'date':'2026-11-06','auditor':'gpt-6-astra-ultra',
 'overall_verdict':'warn','integrity_status':'warn','blocking_findings':[], 'checks':checks,'findings':findings,'claims':claims,
 'source_identities':{'tested_tree_sha256':publication['tested_tree_sha256'],'publication_tree_sha256':publication['tree_sha256'],'file_count':4075,'executable_AST_difference':False},
 'verified_counts':{'selected_suite':counts,'selected_files':21,'targeted_cases':26,'registered_autograd_cases':22,'registered_precision_cases':10,'final_training_ratios_recomputed':448},
 'training_metrics':{mode:{kind:{name:value['total']['relative_l2'] for name,value in training[mode]['metrics'][kind].items()} for kind in ('gradients','updates')} for mode in training},
 'review_limits':['Local source and aggregate evidence only; raw tensors not independently recomputed.','Same-family provisional reviewer, not cross-family acceptance.','One seed, two inputs and two optimizer steps per precision; cached H100.','Whole-layer Linear is not the default adopted path.','One historical auxiliary clone-response prose was compacted before archival; disclosed restatement retained.'],
 'trace_completeness':{'primary_request_and_final_authored_response':True,'auxiliary_historical_gap':True,'gap_details':'Call 006 preserves exact recovered requests and current full response; original earlier clone-response prose unavailable after compaction.'},
 'numeric_accuracy_pass_asserted':False,'training_timing_valid':False,'new_H100_execution':False,
}
(P/'EXPERIMENT_AUDIT.md').write_text(body)
(P/'EXPERIMENT_AUDIT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
(T/'001-integrity-audit.response.md').write_text(body)
(T/'audited_inputs_final_snapshot.json').write_text(json.dumps(inputs, indent=2)+'\n')
meta = json.loads((T/'001-integrity-audit.meta.json').read_text())
meta.update(status='complete', completed_at=now, verdict='WARN', acceptance_status='provisional',
            response_sha256=sha(T/'001-integrity-audit.response.md'), response_kind='complete authored audit report',
            trace_historical_auxiliary_gap_disclosed=True)
(T/'001-integrity-audit.meta.json').write_text(json.dumps(meta, indent=2)+'\n')
meta = json.loads((T/'run.meta.json').read_text())
meta.update(status='complete', completed_at=now, verdict='WARN', report_sha256=sha(P/'EXPERIMENT_AUDIT.md'),
            json_sha256=sha(P/'EXPERIMENT_AUDIT.json'), trace_historical_auxiliary_gap_disclosed=True)
(T/'run.meta.json').write_text(json.dumps(meta, indent=2)+'\n')
with (T/'events.jsonl').open('a') as stream:
    stream.write(json.dumps({'time':now,'event':'integrity_audit_complete','verdict':'WARN','acceptance_status':'provisional','review_independence':'same-family','report_sha256':sha(P/'EXPERIMENT_AUDIT.md'),'json_sha256':sha(P/'EXPERIMENT_AUDIT.json')})+'\n')
print(json.dumps({'verdict':'WARN','audited_input_count':len(inputs),'report_sha256':sha(P/'EXPERIMENT_AUDIT.md'),'json_sha256':sha(P/'EXPERIMENT_AUDIT.json'),'ratios_recomputed':ratios[0]}))
