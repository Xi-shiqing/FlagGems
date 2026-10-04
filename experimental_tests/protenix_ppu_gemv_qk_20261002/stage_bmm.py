"""Isolated experiment: vector results and underfilled reduction tiles use IEEE."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
baseline = root/'candidate'
candidate = root/'bmm_candidate'
assert not candidate.exists()
shutil.copytree(baseline, candidate, ignore=shutil.ignore_patterns('__pycache__', '.git'))
path = candidate/'flag_gems/ops/bmm.py'
before = path.read_text()
tree = ast.parse(before)
selectors = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_allow_bmm_tf32']
assert len(selectors) == 1
selector = selectors[0]
replacement = '''def _allow_bmm_tf32(batch, M, N, K):
    # Experimental generic policy, not an exact H100 dispatch emulation.
    # A vector result uses IEEE; reductions shorter than a 32-wide tile do too.
    return _allow_tf32() and M > 1 and N > 1 and K >= 32
'''
lines = before.splitlines(keepends=True)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('bmm', 'bmm_out'):
        calls = [x for x in ast.walk(node) if isinstance(x, ast.Call) and
                 isinstance(x.func, ast.Name) and x.func.id == '_allow_bmm_tf32']
        assert len(calls) == 1 and len(calls[0].args) == 3
        line = calls[0].lineno-1
        assert lines[line].count('_allow_bmm_tf32(batch, M, N)') == 1
        lines[line] = lines[line].replace('_allow_bmm_tf32(batch, M, N)', '_allow_bmm_tf32(batch, M, N, K)')
lines[selector.lineno-1:selector.end_lineno] = [replacement]
path.write_text(''.join(lines))
compile(path.read_text(), str(path), 'exec')
(root/'bmm_candidate_sources.json').write_text(json.dumps({
    'bmm_before': hashlib.sha256(before.encode()).hexdigest(),
    'bmm_after': hashlib.sha256(path.read_bytes()).hexdigest(),
    'scope': 'Candidate only; K=1 outer product unchanged. K<32 or vector-result BMM IEEE; other FP32 BMM under TF32 flag uses RNE TF32, including narrow PV.'}, indent=2)+'\n')
print('BMM_CANDIDATE_STAGED')
