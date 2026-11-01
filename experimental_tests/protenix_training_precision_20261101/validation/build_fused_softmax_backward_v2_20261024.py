"""New isolated package, retaining the failed v1 sources and evidence."""
from pathlib import Path
source = Path(__file__).with_name('build_fused_softmax_backward_candidate_20261024.py').read_text()
for old, new in {
    'gems_fused_softmax_backward_20261024': 'gems_fused_softmax_backward_v2_20261024',
    'fused_softmax_backward_patch_manifest_20261024.json': 'fused_softmax_backward_v2_manifest_20261024.json',
    'fused_softmax_backward_candidate_20261024.py': 'fused_softmax_backward_v2_candidate_20261024.py',
}.items():
    assert old in source, old
    source = source.replace(old, new)
exec(compile(source, __file__, 'exec'), globals())
