"""Fixed real atomic-module VJPs, current Gems, existing H100 references only.

Keep one precision throughout each forward and backward, including checkpoint
recomputation. No production source, checkpoint or training recipe is changed.
"""
from collections import Counter
import hashlib
import json
import os
import types
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / 'train_backward_profile_20261018'
OUT = Path(sys.argv[sys.argv.index('--output-root') + 1])
sys.path[0] = str(ROOT / 'checkpoint_softmax_stable_v2_r0_20261020')
sys.path.append(str(HERE))  # Diagnostic helper modules; preserve runtime_guard priority.
if os.getenv('PROTENIX_EXTRA_SITE_PACKAGES'):
    sys.path.append(os.environ['PROTENIX_EXTRA_SITE_PACKAGES'])
sys.modules.setdefault('wandb', types.ModuleType('wandb'))
from runtime_guard_long_reduction import verify

imports = verify()
import torch
from training_fixed_state_helpers_v2_20261007ar import file_sha

BASE = OLD / 'entry_replay.py'
assert file_sha(BASE) == 'a5b18c6852d59de30cc88456851b8da9f5f9637b7978178c43f3329ad76bbf70'
fixture_path = OLD / 'h100_sftp_references/more_component_fixtures/component_fixtures.pt'
assert file_sha(fixture_path) == 'fa97504176660987762e5b55a0e59a7c252f07b9e48279e2cfb4b02a7f2301d7'
SELECTED = ['model.template_embedder.pairformer_stack.blocks.0.tri_att_start', 'model.template_embedder.pairformer_stack.blocks.0.tri_mul_out']
policy_state = {'policy': None, 'counts': Counter()}
triangle_modules = {}
triangle_calls = Counter()
rounding_bit_controls = {}
long_reduction_calls = Counter()
original_long_reduction = [None,None]
original_bmm = [None, None]


def install_policy_observer():
    module = sys.modules['flag_gems.ops.bmm']
    assert file_sha(module.__file__) == '2352062170a4b4716f8e8f20419fa2056fd629e0698f34c318445c96ac23cecb'
    original_bmm[:] = [module, module._allow_bmm_tf32]

    def decision(batch, M, N):
        baseline = original_bmm[1](batch, M, N)
        actual = baseline
        key = (policy_state['policy'], batch, M, N, bool(baseline), bool(actual))
        policy_state['counts'][key] += 1
        return actual

    module._allow_bmm_tf32 = decision

    import importlib
    matrix=importlib.import_module('flag_gems.ops.mm')
    original_long_reduction[:]=[matrix,matrix._long_reduction_splitk_mm]
    def long_reduction(a,b,c,M,N,K):
        parts=matrix._long_reduction_splitk_parts(a,b,c,M,N,K)
        value=original_long_reduction[1](a,b,c,M,N,K)
        if value is not None:
            assert parts is not None
            long_reduction_calls[(policy_state['policy'],M,N,K,parts,bool(matrix._allow_tf32()))]+=1
        return value
    matrix._long_reduction_splitk_mm=long_reduction

    import importlib
    from triangle_rounding_rtz_helper import validate_rounding
    candidate = importlib.import_module('triangle_training_rtz_compat')
    manifest_path = HERE / 'prepare_isolated_triangle_RTZ_compat_CPU/triangle_rtz_compat_candidate_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert file_sha(candidate.__file__) == manifest['candidate_sha256']
    rounding_bit_controls.update(validate_rounding())
    namespace = importlib.import_module('flag_gems.ops')
    actual_module = importlib.import_module('flag_gems.ops.triangle_attention')
    original_function = actual_module.triangle_attention
    triangle_modules.update(candidate=candidate, namespace=namespace, actual_module=actual_module, original_function=original_function)
    def selected_triangle(*args, **kwargs):
        policy = policy_state['policy']
        triangle_calls[policy] += 1
        function = candidate.triangle_attention if policy.endswith('rtz_compat') else original_function
        return function(*args, **kwargs)
    namespace.triangle_attention = actual_module.triangle_attention = selected_triangle


source = BASE.read_text()
old = "        model_ref.update(model=model, loss=self.loss, trainer=self)\n"
assert source.count(old) == 1
source = source.replace(old, old + '        install_policy_observer()\n')
old = "        for name, fixture in package['fixtures'].items():\n"
start = source.index(old)
end = source.index("        torch.save({'results': results", start)
block = source[start:end]
block = block.replace(old, "        for name in SELECTED:\n            fixture = package['fixtures'][name]\n", 1)
block = block.replace("            results[name] = row\n", "            results[policy + '/' + str(repeat) + '/' + name] = row\n")
block = ''.join('        ' + line if line.strip() else line for line in block.splitlines(keepends=True))
source = source[:start] + """        for policy in ['fp32_default', 'fp32_rtz_compat', 'tf32_default', 'tf32_rtz_compat']:
            policy_state['policy'] = policy
            torch.backends.cuda.matmul.allow_tf32 = not policy.startswith('fp32')
            os.environ['PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION'] = 'ieee' if policy.startswith('fp32') else 'tf32'
            for repeat in range(2):
""" + block + source[end:]
original_precision = torch.backends.cuda.matmul.allow_tf32
try:
    exec(compile(source, __file__, 'exec'), globals())
finally:
    torch.backends.cuda.matmul.allow_tf32 = original_precision
    if original_bmm[0] is not None:
        original_bmm[0]._allow_bmm_tf32 = original_bmm[1]
    if triangle_modules:
        triangle_modules['namespace'].triangle_attention = triangle_modules['actual_module'].triangle_attention = triangle_modules['original_function']
    evidence = {
        'actual_runtime_imports': imports,
        'actual_long_reduction_calls':[{'policy':key[0],'M_N_K':list(key[1:4]),'split_K':key[4],'TF32':key[5],'count':count} for key,count in long_reduction_calls.items()],
        'rounding_bit_controls': rounding_bit_controls,
        'actual_triangle_policy_calls': dict(triangle_calls),
        'RTZ_candidate_physical_forward_calls': getattr(triangle_modules.get('candidate'), '_PHYSICAL_FORWARD_CALLS', 0),
        'RTZ_candidate_physical_backward_calls': getattr(triangle_modules.get('candidate'), '_PHYSICAL_BACKWARD_CALLS', 0),
        'public_RNE_source_changed': False,
        'helper_evaluations_not_kernel_launches': [
            {'policy': key[0], 'batch_M_N': list(key[1:4]), 'baseline_decision': key[4],
             'actual_decision': key[5], 'count': count}
            for key, count in policy_state['counts'].items()],
        'production_source_changed': False, 'new_H100_execution': False, 'timing_valid': False,
    }
    with (OUT / 'long_reduction_fixed_VJP_execution.json').open('x') as handle:
        json.dump(evidence, handle, indent=2)

# The original Gems scope has unwound. Compare complete saved CPU vectors.
torch.set_num_threads(2)
saved = torch.load(OUT / 'component_replays.pt', map_location='cpu', mmap=True, weights_only=False)
assert saved['fixture_sha256'] == file_sha(fixture_path)
refs = {}
expected = {
    'fp32': '8f9335a40ebd696c149d99692e54b4881ba17a1265c05a69266b5408f361da87',
    'tf32': '7893c4ec070e648985071e29b608ab51bcbeffbd647dbf74110ad8086ffd27be',
}
for precision, sha in expected.items():
    path = OLD / 'h100_sftp_references' / (precision + '_more_hfp') / 'component_replays.pt'
    assert file_sha(path) == sha
    refs[precision] = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert refs[precision]['fixture_sha256'] == saved['fixture_sha256']
    assert refs[precision]['model_initial_sha256'] == saved['model_initial_sha256']


def vector_metric(actual, expected):
    assert actual.keys() == expected.keys()
    e2 = r2 = a2 = 0.
    count = changed = 0
    maximum = 0.
    for name, value in actual.items():
        ref = expected[name]
        assert value.shape == ref.shape and value.dtype == ref.dtype
        av, bv = value.reshape(-1), ref.reshape(-1)
        for start in range(0, av.numel(), 1048576):
            a, b = av[start:start + 1048576].double(), bv[start:start + 1048576].double()
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            delta = a - b
            e2 += float(delta.square().sum())
            r2 += float(b.square().sum())
            a2 += float(a.square().sum())
            changed += int(torch.count_nonzero(delta))
            maximum = max(maximum, float(delta.abs().max()) if delta.numel() else 0.)
            count += a.numel()
    return {'relative_l2': (e2 / max(r2, 1e-300)) ** .5, 'difference_squared': e2,
            'reference_squared': r2, 'actual_squared': a2, 'elements': count,
            'different_elements': changed, 'max_abs': maximum, 'bitwise_equal': changed == 0}


def compare_rows(a, b):
    return {
        'outputs': vector_metric(dict(walk_cpu(a['outputs'])), dict(walk_cpu(b['outputs']))),
        'input_gradients': vector_metric(a['input_gradients'], b['input_gradients']),
        'parameter_gradients': vector_metric(a['parameter_gradients'], b['parameter_gradients']),
    }


def walk_cpu(value, path=''):
    if torch.is_tensor(value):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from walk_cpu(child, path + '/' + str(key))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from walk_cpu(child, path + '/' + str(index))


summary = {
    'status': 'COMPLETE_GENERIC_LONG_REDUCTION_FIXED_REAL_VJP_COMPARISON',
    'fixture_sha256': saved['fixture_sha256'], 'model_initial_sha256': saved['model_initial_sha256'],
    'existing_H100_reference_sha256': expected, 'components': {}, 'new_H100_execution': False,
    'production_source_changed': False, 'timing_valid': False,
    'scope': 'Template TriAttention and a TriMulOut negative control, fixed common real H100 FP32 activations and upstream gradients. Four PPU policies IEEE/RNE/experimental legacy RTZ, two repeats each, complete outputs/input/parameter gradients. Public RNE unchanged; no emulation of H inference CUE query bug. Template shape does not substitute for the full main Pairformer. Not optimizer or checkpoint acceptance.',
}
for name in SELECTED:
    item = {'policies': {}}
    for policy in ['fp32_default', 'fp32_rtz_compat', 'tf32_default', 'tf32_rtz_compat']:
        a, b = [saved['results'][policy + '/' + str(repeat) + '/' + name] for repeat in range(2)]
        repeat = compare_rows(a, b)
        item['policies'][policy] = {
            'two_repeats': repeat,
            'vs_H100_FP32': compare_rows(a, refs['fp32']['results'][name]),
            'vs_H100_TF32': compare_rows(a, refs['tf32']['results'][name]),
        }
    summary['components'][name] = item
summary['all_two_repeats_bitwise_equal'] = all(
    metric['bitwise_equal'] for item in summary['components'].values()
    for policy in item['policies'].values() for metric in policy['two_repeats'].values())
assert evidence['RTZ_candidate_physical_forward_calls'] > 0 and evidence['RTZ_candidate_physical_backward_calls'] > 0
summary['same_FP32_complete_vector_controls'] = {}
summary['non_TriAttention_TF32_negative_control'] = {}
for name in SELECTED:
    fp = compare_rows(saved['results']['fp32_rtz_compat/0/' + name], saved['results']['fp32_default/0/' + name])
    summary['same_FP32_complete_vector_controls'][name] = fp
    assert all(value['bitwise_equal'] for value in fp.values()), 'Copied IEEE path changed'
    if name.endswith('tri_mul_out'):
        control = compare_rows(saved['results']['tf32_rtz_compat/0/' + name], saved['results']['tf32_default/0/' + name])
        summary['non_TriAttention_TF32_negative_control'][name] = control
        assert all(value['bitwise_equal'] for value in control.values()), 'Counterfactual changed unrelated operator'
previous_path=HERE/'fixed_real_triangle_RNE_RTZ_compatibility_VJP_v2/component_replays.pt'
previous=torch.load(previous_path,map_location='cpu',mmap=True,weights_only=False)
assert previous['fixture_sha256']==saved['fixture_sha256'] and previous['model_initial_sha256']==saved['model_initial_sha256']
summary['original_Gems_replay_sha256']=file_sha(previous_path)
summary['vs_original_Gems_same_policies']={}
for key,old_row in previous['results'].items():
    compared=compare_rows(saved['results'][key],old_row)
    assert compared['outputs']['bitwise_equal'] and compared['input_gradients']['bitwise_equal'],'Long reduction changed forward or propagated gradient at fixed input'
    summary['vs_original_Gems_same_policies'][key]=compared
assert evidence['actual_long_reduction_calls']
summary['scope']='Isolated FlagGems one-file generic low-occupancy long-reduction MM/mm_out route, unchanged IEEE/RNE semantics; standard RNE and legacy RTZ TriAttention controls. Same real input/upstream gradients; all original-Gems forwards and propagated input gradients must remain bitwise equal. Not a production or model-quality acceptance.'
with (OUT / 'long_reduction_fixed_VJP_comparison.json').open('x') as handle:
    json.dump(summary, handle, indent=2, allow_nan=False)
print(summary['status'], 'repeat_exact', summary['all_two_repeats_bitwise_equal'], flush=True)
