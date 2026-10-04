"""Stage an isolated MM-to-MV candidate on the existing K=1 repair."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
baseline = root.parent/'rank_one_20261002/candidate'
candidate = root/'candidate'
assert not candidate.exists()
shutil.copytree(baseline, candidate, ignore=shutil.ignore_patterns('__pycache__', '.git'))
ops = candidate/'flag_gems/ops'
shutil.copy2(root/'mm_gemv.py', ops/'mm_gemv.py')
path = ops/'mm.py'
before = path.read_text()
lines = before.splitlines(keepends=True)
changes = []
for node in ast.parse(before).body:
    if isinstance(node, ast.FunctionDef) and node.name in ('mm', 'mm_out'):
        anchors = [n for n in node.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Tuple) and any(isinstance(x, ast.Name) and x.id == 'N' for x in t.elts) for t in n.targets)]
        assert len(anchors) == 1
        output = 'out' if node.name == 'mm_out' else 'None'
        changes.append((anchors[0].end_lineno,
            f'    if K > 1 and (M == 1 or N == 1) and _allow_tf32():\n'
            f'        result = mm_gemv(a, b, {output}, enabled=True)\n'
            f'        if result is not None:\n'
            f'            return result\n'))
assert len(changes) == 2
for line, value in sorted(changes, reverse=True):
    lines.insert(line, value)
path.write_text('from flag_gems.ops.mm_gemv import mm_gemv\n'+''.join(lines))
compile(path.read_text(), str(path), 'exec')
(root/'candidate_sources.json').write_text(json.dumps({'mm_before':hashlib.sha256(before.encode()).hexdigest(),
    'mm_after':hashlib.sha256(path.read_bytes()).hexdigest(),
    'helper':hashlib.sha256((ops/'mm_gemv.py').read_bytes()).hexdigest(),
    'policy':'K>1 and M=1 or N=1 only; existing IEEE MV; all FP32 and non-vector paths unchanged.'}, indent=2)+'\n')
print('GEMV_CANDIDATE_STAGED')
