"""Change only reviewed complete-writing allocation sites in a copied package."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    source = args.baseline / 'flag_gems'
    target = args.output / 'flag_gems'
    assert not target.exists(), 'Never overwrite an earlier candidate'
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(Path(__file__).with_name('full_write_allocation.py'),
                    target / 'utils/full_write_allocation.py')
    scopes = {'utils/pointwise_dynamic.py': {'prepare_args'},
              'ops/softmax.py': {'softmax'}}
    expected = {'utils/pointwise_dynamic.py': 3, 'ops/softmax.py': 2}
    if args.extended:
        scopes.update({'ops/sigmoid.py': {'_sigmoid_measured_ppu'},
                       'ops/triangle_attention.py': {'_streaming_triangle_attention_forward_impl'},
                       'ops/cat.py': {'cat'}})
        expected.update({'ops/sigmoid.py': 1, 'ops/triangle_attention.py': 1, 'ops/cat.py': 1})
        shutil.copyfile(Path(__file__).with_name('mm_preserve_layout.py'), target/'ops/mm_preserve_layout.py')
    changes, patch = {}, []
    for name, names in scopes.items():
        path = target / name
        before = path.read_text()
        lines = before.splitlines(keepends=True)
        count = 0
        for node in ast.walk(ast.parse(before)):
            if not isinstance(node, ast.FunctionDef) or node.name not in names:
                continue
            for index in range(node.lineno-1, node.end_lineno):
                line = lines[index]
                count += line.count('torch.empty_like(')
                lines[index] = line.replace('torch.empty_like(', '_full_write.empty_like(')
                if name != 'ops/triangle_attention.py':
                    count += line.count('torch.empty(')
                    lines[index] = lines[index].replace('torch.empty(', '_full_write.empty(')
        assert count == expected[name], (name, count)
        # Keep module docstrings and future imports at the start of the module.
        tree = ast.parse(before)
        insertion = 0
        for node in tree.body:
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)) or (
                    isinstance(node, ast.ImportFrom) and node.module == '__future__'):
                insertion = node.end_lineno
            else:
                break
        lines.insert(insertion, 'from flag_gems.utils import full_write_allocation as _full_write\n')
        after = ''.join(lines)
        compile(after, str(path), 'exec')
        path.write_text(after)
        changes[name] = {'allocation_sites': count,
                        'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
                        'after_sha256': hashlib.sha256(after.encode()).hexdigest()}
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile='a/flag_gems/'+name, tofile='b/flag_gems/'+name))
    prefix = args.output.name
    (args.output.parent/(prefix+'.patch')).write_text(''.join(patch))
    (args.output.parent/(prefix+'_manifest.json')).write_text(json.dumps(changes, indent=2)+'\n')
    print('FULL_WRITE_CANDIDATE', changes)


if __name__ == '__main__':
    main()
