"""Insert a shared rank-one branch into the installed, frozen experiment sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def patch(source, module, *, frozen=True):
    tree = ast.parse(source)
    changes = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in {
            'mm', 'mm_out', 'addmm', 'addmm_out', 'bmm', 'bmm_out', '_addmm_impl'}:
            continue
        anchors = [s for s in node.body if isinstance(s, ast.Assign) and any(
            isinstance(t, ast.Tuple) and any(isinstance(x, ast.Name) and x.id == 'N' for x in t.elts)
            for t in s.targets)]
        assert len(anchors) == 1, node.name
        a, b = ('A', 'B') if module == 'bmm' else (('mat1', 'mat2') if module == 'addmm' else ('a', 'b'))
        out = 'out' if node.name.endswith('_out') or node.name == '_addmm_impl' else 'None'
        enabled = '_allow_tf32()' if frozen else '(runtime.device.vendor_name == "thead" and torch.backends.cuda.matmul.allow_tf32)'
        extra = ', bias=bias, alpha=alpha, beta=beta' if module == 'addmm' else ''
        code = f'    if K == 1 and {enabled}:\n        result = rank_one({a}, {b}, {out}, enabled=True{extra})\n        if result is not None:\n            return result\n'
        changes.append((anchors[0].end_lineno, code))
    assert changes, module
    lines = source.splitlines(keepends=True)
    for line, code in sorted(changes, reverse=True):
        lines.insert(line, code)
    result = 'from flag_gems.ops.rank_one import rank_one\n'+''.join(lines)
    compile(result, module, 'exec')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--helper', type=Path, required=True)
    args = p.parse_args()
    assert not args.destination.exists(), 'Use a new isolated candidate; never overwrite prior results'
    shutil.copytree(args.baseline, args.destination, ignore=shutil.ignore_patterns('__pycache__', '.git'))
    ops = args.destination/'flag_gems/ops'
    shutil.copy2(args.helper, ops/'rank_one.py')
    hashes = {}
    for name in ('mm', 'addmm', 'bmm'):
        path = ops/(name+'.py')
        before = path.read_text()
        path.write_text(patch(before, name))
        hashes[name] = {'before': hashlib.sha256(before.encode()).hexdigest(),
                        'after': hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.destination.parent/'candidate_sources.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print('CANDIDATE_STAGED', hashes)


if __name__ == '__main__':
    main()
