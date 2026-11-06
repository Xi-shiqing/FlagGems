"""Unchanged real two-step recipe with the complete merged FlagGems source package."""
import hashlib
import json
from pathlib import Path
import sys

CURRENT = Path(__file__).resolve().parent
OLD = Path('/mnt/cpfs/xishiqing/ppu_full_acceptance_20261005/precision_residual_20261019')
PACKAGE = CURRENT/'packages/merged_complete'
manifest = json.loads((CURRENT/'merged_complete_all_source_hashes.json').read_text())
sys.path.insert(0, str(CURRENT))
source = (OLD/'train_softmax_forward_fidelity_20261023.py').read_text()
changes = {
    'HERE=Path(__file__).resolve().parent': 'HERE=Path('+repr(str(OLD))+')',
    'from softmax_forward_fidelity_mode_20261023 import SoftmaxForwardMode':
        'from upstream_training_observer_20261106 import MergedTrainingObserver as SoftmaxForwardMode',
    'PROTENIX_SOFTMAX_FORWARD_FIDELITY': 'PROTENIX_UPSTREAM_MERGE_RULE',
    "assert path.is_relative_to(HERE/'gems_canonical_softmax_v2')":
        'assert path.is_relative_to(Path('+repr(str(PACKAGE))+')/"flag_gems")',
    "'provider_manifest':json.loads((HERE/'softmax_fidelity_provider_manifest_20261023.json').read_text()) if kind=='accurate' else None,":
        "'registration_observation':mode.evidence(),",
    'softmax_forward_fidelity_mode_20261023.py': str(CURRENT/'upstream_training_observer_20261106.py'),
    'softmax_forward_training_execution.json': 'upstream_merge_training_execution.json',
    'All actual CUDA FP32 aten._softmax forward entries. Original ATen autograd and Gems backward retained; fused attention internal Softmax is outside this scope. TF32 matrix precision unchanged.':
        'Complete upstream-merged FlagGems package with current accepted PPU FP32/RNE matrix kernels and fused Softmax retained as backend providers. Same model, data, RNG, precision configuration and optimizer.',
}
original = {
    'mm': '5e1cf91e8b62929dd2d96990d7b2232f539fdd7a0e6c0d5ed79a0a45d551e7b8',
    'bmm': '2352062170a4b4716f8e8f20419fa2056fd629e0698f34c318445c96ac23cecb',
    'softmax': '2148bd988c577db8bd6aa01dcf6a7bfe37722d03d12bd7db9829f2199e9b1a10',
    'index_put': 'e556112b97d08ab6829b51875742b65cec151641614fdc6801596d2350a26a42',
    'unfold_backward': 'aa86a32917aeee7a307341c47c7114ab2e6eba2d81594f1453896f0bfb3d4777',
}
for name, old in original.items():
    changes[old] = manifest['flag_gems/ops/'+name+'.py']
for old, new in changes.items():
    assert old in source, old
    source = source.replace(old, new)
exec(compile(source, __file__, 'exec'), globals())
