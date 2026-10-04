"""Reuse the tested RNE kernels; write gradients directly into bounded chunks."""
import ast
import hashlib
import json
from pathlib import Path

import candidate_build_base as base

ROOT = Path(__file__).resolve().parent

WRAPPER = '''

from PARENT import _triangle_attention_backward as _parent_backward

_INPUT_CACHE_BUDGET_BYTES = BUDGET

def _triangle_attention_backward(q, k, v, bias1, bias2, out, lse, do, precision):
    if q.dtype != torch.float32 or precision not in {"tf32", "tf32_rne"}:
        return _parent_backward(q, k, v, bias1, bias2, out, lse, do, precision)
    b, n, s, h, d = q.shape
    bytes_per_group = 4 * s * h * d * q.element_size()
    if bytes_per_group > _INPUT_CACHE_BUDGET_BYTES:
        return _parent_backward(q, k, v, bias1, bias2, out, lse, do, precision)
    group_count = min(n, _INPUT_CACHE_BUDGET_BYTES // bytes_per_group)
    workspace = torch.empty((4 * group_count * s * h * d,), dtype=q.dtype, device=q.device)
    destinations = [torch.empty(q.shape, dtype=q.dtype, device=q.device) for _ in range(3)]
    if b == 1 and group_count == n:
        return _direct_chunk_backward(q, k, v, bias1, bias2, out, lse, do, precision,
                                      converted_workspace=workspace, grad_destinations=destinations)
    dbias = torch.zeros(bias2.shape, dtype=bias2.dtype, device=bias2.device)
    for batch in range(b):
        for start in range(0, n, group_count):
            index = (slice(batch, batch + 1), slice(start, min(n, start + group_count)))
            operands = [value[index] for value in (q, k, v, bias1)]
            gradients = _direct_chunk_backward(*operands, bias2[batch:batch+1], out[index], lse[index], do[index], precision,
                converted_workspace=workspace, grad_destinations=[value[index] for value in destinations])
            dbias[batch:batch+1].add_(gradients[3])
            del gradients
    return *destinations, dbias
'''


def combined(source, prefix, budget):
    source = base.prequant(source)
    host = source.index('def _triangle_attention_backward(')
    signature_end = source.index(') -> tuple[', host)
    source = source[:signature_end] + '    *,\n    converted_workspace=None,\n    grad_destinations=None,\n' + source[signature_end:]
    source = source.replace('def _triangle_attention_backward(', 'def _direct_chunk_backward(', 1)
    allocation = '        rounded = torch.empty((4, size), dtype=q.dtype, device=q.device)'
    assert source.count(allocation) == 1
    source = source.replace(allocation, '        rounded = converted_workspace[:4*size].view(4, size)')
    outputs = '    dq = torch.empty_like(q)\n    dk = torch.empty_like(k)\n    dv = torch.empty_like(v)'
    assert source.count(outputs) == 1
    source = source.replace(outputs, '    dq, dk, dv = grad_destinations\n    assert all(x.shape == q.shape and x.dtype == q.dtype and x.device == q.device and x.is_contiguous() for x in (dq, dk, dv))')
    source = base.forward_kv(source)
    assert source.count('def _round_four_inputs(') == 2
    assert source.endswith(base.PREQUANT_KERNEL)
    source = source[:-len(base.PREQUANT_KERNEL)]
    source += WRAPPER.replace('PARENT', prefix + 'fixed_rne_once').replace(' = BUDGET', ' = ' + str(budget), 1)
    ast.parse(source)
    return source


def main():
    manifest = {'scope': 'Generic byte-budget chunks; reused converted workspace; kernel writes final gradient slices directly. Raw dO preserved for delta. No shape whitelist or changed default package.', 'candidates': {}}
    for prefix in ('', 'cuda_'):
        parent = ROOT / (prefix + 'fixed_rne_once.py')
        for mib in (64, 128, 256):
            name = prefix + f'combined_{mib}m'
            path = ROOT / (name + '.py')
            path.write_text(combined(parent.read_text(), prefix, mib * 1024**2))
            manifest['candidates'][name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(), 'converted_workspace_budget_bytes': mib * 1024**2}
    (ROOT / 'candidate_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('COMBINED_DIRECT_DESTINATION_CANDIDATES_READY', list(manifest['candidates']))


if __name__ == '__main__':
    main()
