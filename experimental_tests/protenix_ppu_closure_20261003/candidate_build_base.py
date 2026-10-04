"""Isolated backward dataflow experiments from the previously tested kernels."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUDGET = 256 * 1024**2

PREQUANT_KERNEL = '''

@triton.jit
def _round_four_inputs(q, k, v, do, destination, size: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    role = tl.program_id(1)
    if role == 0:
        source = q
    elif role == 1:
        source = k
    elif role == 2:
        source = v
    else:
        source = do
    value = tl.load(source + index, index < size, other=0.)
    value = round_tf32_rne(value, True)
    tl.store(destination + role * size + index, value, index < size)
'''

CHUNK_WRAPPER = '''

_DS_CACHE_BUDGET_BYTES = 256 * 1024**2

def _triangle_attention_backward(q, k, v, bias1, bias2, out, lse, do, precision):
    """Stream N groups through bounded FP32 dS storage, with deterministic bias sums."""
    if q.dtype != torch.float32 or precision not in {"tf32", "tf32_rne"}:
        return _cached_chunk_backward(q, k, v, bias1, bias2, out, lse, do, precision)
    b, n, s, h, d = q.shape
    bytes_per_group = h * s * s * 4
    if bytes_per_group > _DS_CACHE_BUDGET_BYTES:
        return _cached_chunk_backward(q, k, v, bias1, bias2, out, lse, do, precision)
    group_count = min(n, _DS_CACHE_BUDGET_BYTES // bytes_per_group)
    if b == 1 and group_count == n:
        return _cached_chunk_backward(q, k, v, bias1, bias2, out, lse, do, precision)
    dq, dk, dv = [torch.empty_like(value) for value in (q, k, v)]
    dbias = torch.zeros_like(bias2)
    for batch in range(b):
        for start in range(0, n, group_count):
            stop = min(n, start + group_count)
            index = (slice(batch, batch + 1), slice(start, stop))
            operands = [value[index] for value in (q, k, v, bias1)]
            gradients = _cached_chunk_backward(*operands, bias2[batch:batch+1], out[index], lse[index], do[index], precision)
            for destination, gradient in zip((dq, dk, dv), gradients[:3]):
                destination[index].copy_(gradient)
            dbias[batch:batch+1].add_(gradients[3])
    return dq, dk, dv, dbias
'''


def backward_region(source):
    first = source.index('def _streaming_triangle_attention_bwd_dq_v1(')
    last = source.index('\ndef _validate_precision(', first)
    return first, last, source[first:last]


def prequant(source):
    first, last, kernels = backward_region(source)
    for value in ('q', 'k', 'v', 'do'):
        for role in ('qk', 'pv'):
            kernels = kernels.replace(f'round_tf32_rne({value}, {role}_precision_mode == 5)', value)
    # Raw upstream must still produce delta=sum(out*dO), never the quantized dO.
    dq_end = kernels.index('\n\n@triton.jit')
    dq = kernels[:dq_end].replace('    do_ptr,\n', '    do_ptr,\n    do_delta_ptr,\n', 1)
    dq = dq.replace('    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)',
                    '    do_delta = tl.load(do_delta_ptr + q_offsets, mask=valid_m[:, None], other=0.0)\n    delta = tl.sum(out.to(tl.float32) * do_delta.to(tl.float32), axis=1)')
    source = source[:first] + dq + kernels[dq_end:] + source[last:]
    host = source.index('def _triangle_attention_backward(')
    anchor = source.index('    dq = torch.empty_like(q)\n', host)
    source = source[:anchor] + '''    do_raw = do
    if precision in {"tf32", "tf32_rne"} and q.dtype == torch.float32:
        size = q.numel()
        rounded = torch.empty((4, size), dtype=q.dtype, device=q.device)
        _round_four_inputs[(triton.cdiv(size, 256), 4)](q, k, v, do, rounded, size, 256, num_warps=4)
        q, k, v, do = [rounded[index].view(q.shape) for index in range(4)]
    elif precision != "ieee":
        raise ValueError("prequant diagnostic supports IEEE and full RNE TF32 only")
''' + source[anchor:]
    source = source.replace('        do,\n        ds_scratch,\n', '        do,\n        do_raw,\n        ds_scratch,\n', 1)
    return source + PREQUANT_KERNEL


def cache_ds(source):
    old = '''    use_fused_dbias_scratch = (
        q.dtype == torch.float32
        and (b, n, s, d) == (1, 693, 693, 32)
        and h in {2, 4, 8}
        and tf32_mode == "none"
    )
    use_split_dkdv = use_fused_dbias_scratch and h == 8'''
    new = '''    legacy_fp32_scratch = (
        q.dtype == torch.float32
        and (b, n, s, d) == (1, 693, 693, 32)
        and h in {2, 4, 8}
        and tf32_mode == "none"
    )
    generic_tf32_scratch = (
        q.dtype == torch.float32
        and tf32_mode == "full_rne"
        and b * n * h * s * s * 4 <= _DS_CACHE_BUDGET_BYTES
    )
    use_fused_dbias_scratch = legacy_fp32_scratch or generic_tf32_scratch
    use_split_dkdv = use_fused_dbias_scratch and (generic_tf32_scratch or h == 8)'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    source = source.replace('torch.float16 if h == 8 and use_fp16_training_scratch else torch.float32',
                            'torch.float16 if legacy_fp32_scratch and h == 8 and use_fp16_training_scratch else torch.float32')
    source = source.replace('def _triangle_attention_backward(', 'def _cached_chunk_backward(', 1)
    return source + CHUNK_WRAPPER


def bounded_prequant(source, prefix):
    source = prequant(source).replace('def _triangle_attention_backward(', 'def _prequant_chunk_backward(', 1)
    wrapper = CHUNK_WRAPPER.replace('_DS_CACHE_BUDGET_BYTES', '_INPUT_CACHE_BUDGET_BYTES')
    wrapper = wrapper.replace('h * s * s * 4', '4 * s * h * d * 4')
    wrapper = wrapper.replace('_cached_chunk_backward', '_prequant_chunk_backward')
    wrapper = wrapper.replace('return _prequant_chunk_backward(q, k, v, bias1, bias2, out, lse, do, precision)',
                              'return _parent_backward(q, k, v, bias1, bias2, out, lse, do, precision)', 2)
    wrapper = wrapper.replace('"""Stream N groups through bounded FP32 dS storage, with deterministic bias sums."""',
                              '"""Bound converted Q/K/V/dO bytes and preserve raw dO for delta."""')
    wrapper = '\nfrom ' + prefix + 'fixed_rne_once import _triangle_attention_backward as _parent_backward\n' + wrapper
    return source + wrapper


def forward_kv(source):
    first = source.index('def _streaming_triangle_attention_fwd_ieee_v1(')
    last = source.index('\n\n@triton.jit', first)
    kernel = source[first:last]
    kernel = kernel.replace('    bias_seeded_qk: tl.constexpr,', '    bias_seeded_qk: tl.constexpr,\n    kv_prequantized: tl.constexpr,')
    assert 'kv_prequantized: tl.constexpr' in kernel
    kernel = kernel.replace('round_tf32_rne(k, qk_allow_tf32 == 5)', 'round_tf32_rne(k, qk_allow_tf32 == 5 and not kv_prequantized)')
    kernel = kernel.replace('round_tf32_rne(v, pv_allow_tf32 == 5)', 'round_tf32_rne(v, pv_allow_tf32 == 5 and not kv_prequantized)')
    source = source[:first] + kernel + source[last:]
    host = source.index('def _streaming_triangle_attention_forward_impl(')
    anchor = source.index('    bias_seeded_qk = True\n', host)
    source = source[:anchor] + '''    kv_prequantized = tf32_mode == "full_rne" and 2*q.numel()*q.element_size() <= 256*1024**2
    if kv_prequantized:
        size = q.numel()
        kv = torch.empty((2, size), dtype=q.dtype, device=q.device)
        _round_four_inputs[(triton.cdiv(size, 256), 2)](k, v, k, v, kv, size, 256, num_warps=4)
        k, v = [kv[index].view(q.shape) for index in range(2)]
''' + source[anchor:]
    source = source.replace('        bias_seeded_qk=bias_seeded_qk,', '        bias_seeded_qk=bias_seeded_qk,\n        kv_prequantized=kv_prequantized,', 1)
    return source + PREQUANT_KERNEL


def main():
    manifest = {'status': 'ISOLATED_CANDIDATES_GENERATED', 'ds_cache_budget_bytes': BUDGET,
                'scope': 'No default package or production implementation modified. Generic N streaming by explicit byte budget, no new shape/model whitelist. No FP16 dS cache.', 'candidates': {}}
    for prefix in ('', 'cuda_'):
        source_path = ROOT / (prefix + 'fixed_rne_once.py')
        source = source_path.read_text()
        first, last, kernels = backward_region(source)
        no_round = kernels.replace('qk_precision_mode == 5)', 'False)').replace('pv_precision_mode == 5)', 'False)')
        variants = {'round_off_diagnostic': source[:first] + no_round + source[last:],
                    'prequant_inputs': prequant(source), 'cached_ds': cache_ds(source),
                    'bounded_prequant': bounded_prequant(source, prefix),
                    'prequant_cached_ds': cache_ds(prequant(source)),
                    'forward_kv_prequant': forward_kv(source)}
        for name, text in variants.items():
            ast.parse(text)
            path = ROOT / (prefix + name + '.py')
            path.write_text(text)
            manifest['candidates'][prefix + name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'parent_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                'kind': 'precision-changing cost ablation; never a promotion candidate' if name == 'round_off_diagnostic' else 'semantics-preserving dataflow candidate; requires empirical accuracy validation'}
    (ROOT / 'candidate_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('BACKWARD_CANDIDATES_GENERATED', list(manifest['candidates']))


if __name__ == '__main__':
    main()
