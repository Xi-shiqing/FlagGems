"""Copy current Gems; add one general low-occupancy long-reduction MM route."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--output-root', type=Path, required=True)
args = parser.parse_args()
base = HERE / 'gems_canonical_softmax_v2'
target = HERE / 'gems_long_reduction_splitk'
source_path = base / 'flag_gems/ops/mm.py'
original = source_path.read_text()
assert hashlib.sha256(original.encode()).hexdigest() == '5e1cf91e8b62929dd2d96990d7b2232f539fdd7a0e6c0d5ed79a0a45d551e7b8'
helper = '''

def _long_reduction_splitk_parts(a, b, c, M, N, K):
    """Bounded split-K from output occupancy and reduction work, no shape list."""
    if (not _PPU_PACK_RHS_AVAILABLE
            or os.environ.get("FLAG_GEMS_PPU_LONG_REDUCTION_SPLITK", "1") != "1"
            or a.dtype is not torch.float32 or b.dtype is not torch.float32
            or a.device.type != "cuda" or K <= 8192 or M <= 0 or N <= 0
            or (c is not None and c.dtype is not torch.float32)):
        return None
    for tensor, rows, cols in ((a, M, K), (b, K, N)):
        if ((rows - 1) * abs(tensor.stride(0))
                + (cols - 1) * abs(tensor.stride(1))) > 2147483647:
            return None
    if c is not None and ((M - 1) * abs(c.stride(0))
                          + (N - 1) * abs(c.stride(1))) > 2147483647:
        return None
    output_blocks = triton.cdiv(M, 32) * triton.cdiv(N, 32)
    desired = triton.cdiv(2 * get_sm_count(), output_blocks)
    # At least512 K values per split, at most64 partials and64MiB workspace.
    maximum = min(64, K // 512, (64 * 1024 * 1024) // (4 * M * N))
    if desired <= 1 or maximum < 2:
        return None
    return min(triton.next_power_of_2(desired), 1 << (maximum.bit_length() - 1))


def _long_reduction_splitk_mm(a, b, c, M, N, K):
    parts = _long_reduction_splitk_parts(a, b, c, M, N, K)
    if parts is None:
        return None
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    partials = torch.empty((parts, M, N), device=a.device, dtype=torch.float32)
    with torch_device_fn.device(a.device):
        mm_kernel_h100_fp32_splitk[(triton.cdiv(M, 32) * triton.cdiv(N, 32), parts)](
            a, b, partials, M, N, K, *a.stride(), *b.stride(),
            SPLIT_K=parts, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
            ALLOW_TF32=_allow_tf32(), num_warps=4, num_stages=2)
        mm_kernel_h100_fp32_splitk_reduce[(triton.cdiv(M * N, 256),)](
            partials, c, M * N, *c.stride(), N, parts, BLOCK=256, num_warps=4)
    return c

'''
anchor = '\n_ordered_datatypes = '
assert original.count(anchor) == 1
modified = original.replace(anchor, helper + anchor)
# Both public MM entry points have exactly the same validated M/K/N extraction.
anchor = '    M, K = a.shape\n    _, N = b.shape\n'
assert modified.count(anchor) == 2
for function, output in [('mm', 'None'), ('mm_out', 'out')]:
    tree = ast.parse(modified)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    body = ast.get_source_segment(modified, node)
    assert body.count(anchor) == 1
    replacement = anchor + f'    reduced = _long_reduction_splitk_mm(a, b, {output}, M, N, K)\n    if reduced is not None:\n        return reduced\n'
    modified = modified.replace(body, body.replace(anchor, replacement), 1)
ast.parse(modified)
if not target.exists():
    shutil.copytree(base, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
path = target / 'flag_gems/ops/mm.py'
assert path.read_text() in (original, modified)
path.write_text(modified)
differences = []
for p in base.rglob('*.py'):
    q = target / p.relative_to(base)
    if p.read_bytes() != q.read_bytes():
        differences.append(str(p.relative_to(base)))
assert differences == ['flag_gems/ops/mm.py']
result = {'status': 'COMPLETE_ISOLATED_GENERIC_LONG_REDUCTION_SPLITK_CANDIDATE',
          'gems_path': str(target / 'flag_gems'), 'changed_files': differences,
          'mm_original_sha256': hashlib.sha256(original.encode()).hexdigest(),
          'mm_candidate_sha256': hashlib.sha256(modified.encode()).hexdigest(),
          'package_python_files': len(list(target.rglob('*.py'))),
          'production_source_changed': False, 'new_H100_execution': False,
          'scope': 'All float32 MM/mm_out with long K and too few output CTAs; split count from SM occupancy and bounded workspace. Both IEEE and RNE reuse unchanged existing kernels. No input/model/shape allowlist, autograd rewrite, TF32-to-IEEE fallback, atomic output accumulation or new vendor build. Not yet production acceptance.'}
(args.output_root / 'long_reduction_splitk_candidate_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
print(result['status'], flush=True)
