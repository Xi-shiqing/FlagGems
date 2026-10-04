#!/usr/bin/env python3
"""Avoid redundant initialization of buffers completely written by TF32 kernels."""
import argparse
import difflib
import hashlib
from pathlib import Path
import shutil


HELPER = '''
def empty_full_write(shape, *, dtype, device):
    """Allocate fresh contiguous storage for a kernel that writes every element.

    No global deterministic flag is changed. Callers must never read the buffer
    before the complete-writing kernel, including masked logical edge elements.
    """
    import math
    out = torch.empty(0, dtype=dtype, device=device)
    storage = torch.UntypedStorage(math.prod(shape) * out.element_size(), device=out.device)
    return out.set_(storage, 0, shape)

'''


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--patch', type=Path, required=True)
    parser.add_argument('--tf32-unsplit-stages', type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    root = args.baseline / 'flag_gems'
    assert hashlib.sha256((root / 'ops/mm.py').read_bytes()).hexdigest() == 'fe8686e66bab01824f5bfe28eba019ba9628a436cae5406ed6a65c3f987e7205'
    changes = {}
    for name in ('mm', 'bmm'):
        old = (root / f'ops/{name}.py').read_text()
        new = replace_once(old, '    capture_real_operands,', '    capture_real_operands,\n    empty_full_write,')
        if name == 'mm':
            if args.tf32_unsplit_stages:
                new = replace_once(new,
                    '    split_k = _ppu_h100_splitk_parts(M, N, K) or 1',
                    '    split_k = 1 if _allow_tf32() else (_ppu_h100_splitk_parts(M, N, K) or 1)')
                new = replace_once(new,
                    '            num_warps=4, num_stages=1,\n        )\n    return c',
                    f'            num_warps=4, num_stages={args.tf32_unsplit_stages} if allow_tf32 else 1,\n        )\n    return c')
            new = replace_once(new,
                '    c = torch.empty((M, N), device=device, dtype=c_dtype)',
                '    c = (empty_full_write((M, N), device=device, dtype=c_dtype)\n'
                '         if use_direct and _allow_tf32()\n'
                '         else torch.empty((M, N), device=device, dtype=c_dtype))')
        else:
            new = replace_once(new, '    rounded_a = torch.empty(', '    rounded_a = empty_full_write(')
            new = replace_once(new, '    rounded_b = torch.empty(', '    rounded_b = empty_full_write(')
            new = replace_once(new,
                '    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)\n'
                '    allow_tf32 = _allow_bmm_tf32(batch, M, N)',
                '    allow_tf32 = _allow_bmm_tf32(batch, M, N)\n'
                '    out = (empty_full_write((batch, M, N), dtype=A.dtype, device=A.device)\n'
                '           if allow_tf32 and A.dtype == torch.float32\n'
                '           else torch.empty((batch, M, N), dtype=A.dtype, device=A.device))')
        changes[f'ops/{name}.py'] = (old, new)
    old = (root / 'ops/tf32_precision.py').read_text()
    changes['ops/tf32_precision.py'] = (old, old + HELPER)
    assert not args.candidate.exists(), 'Use a fresh candidate directory'
    shutil.copytree(root, args.candidate / 'flag_gems', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    patches = []
    for name, (old, new) in changes.items():
        (args.candidate / 'flag_gems' / name).write_text(new)
        patches.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                       fromfile='a/flag_gems/' + name, tofile='b/flag_gems/' + name))
    args.patch.write_text(''.join(patches))
    print('FULL_WRITE_CANDIDATE', args.candidate)


if __name__ == '__main__':
    main()
