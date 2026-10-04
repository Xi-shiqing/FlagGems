"""Generate isolated candidates from the actual GPU-tested FlagGems snapshot."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    source = (ROOT / 'frozen_snapshot/triangle_attention.py').read_text()
    helper = (ROOT / 'frozen_snapshot/tf32_precision.py').read_text()
    (ROOT / 'frozen_precision.py').write_text(helper)
    candidates = {}
    for name, function in (
        ('fixed_rne_once', 'round_tf32_rne'),
        ('fixed_rne_bits_once', 'round_tf32_rne_bits'),
        ('fixed_rna_bits_once', 'round_tf32_rna_bits'),
    ):
        result = source.replace('from flag_gems.ops.tf32_precision import (', 'from frozen_precision import (')
        if function != 'round_tf32_rne':
            result = result.replace('    round_tf32_rne,\n', '    ' + function + ',\n')
            result = result.replace('round_tf32_rne(', function + '(')
        if function == 'round_tf32_rna_bits':
            result = result.replace('_rne"', '_rna"')
        first = result.index('def _streaming_triangle_attention_fwd_ieee_v1(')
        last = result.index('\n\n@triton.jit', first)
        kernel = result[first:last]
        operation = function + '(q, qk_allow_tf32 == 5)'
        assert kernel.count(operation) == 2
        kernel = kernel.replace(operation, 'q')
        load = '    q = tl.load(q_ptr + q_offsets, mask=valid_m[:, None], other=0.0)\n'
        assert kernel.count(load) == 1
        kernel = kernel.replace(load, load + '    q = ' + operation + '\n')
        result = result[:first] + kernel + result[last:]
        ast.parse(result)
        path = ROOT / (name + '.py')
        path.write_text(result)
        candidates[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'rounding_helper': function, 'scope': 'Same kernels and schedules; immutable Q converted before KV loop. Backward uses the same selected rounding helper.'}
    helper += '''

@triton.jit
def round_tf32_rne_bits(x, enabled: tl.constexpr):
    if enabled and x.dtype == tl.float32:
        bits = x.to(tl.uint32, bitcast=True)
        finite = (bits & 0x7f800000) != 0x7f800000
        rounded = (bits + 0xfff + ((bits >> 13) & 1)) & 0xffffe000
        return tl.where(finite, rounded, bits).to(tl.float32, bitcast=True)
    return x


@triton.jit
def round_tf32_rna_bits(x, enabled: tl.constexpr):
    if enabled and x.dtype == tl.float32:
        bits = x.to(tl.uint32, bitcast=True)
        finite = (bits & 0x7f800000) != 0x7f800000
        rounded = (bits + 0x1000) & 0xffffe000
        return tl.where(finite, rounded, bits).to(tl.float32, bitcast=True)
    return x
'''
    ast.parse(helper)
    (ROOT / 'frozen_precision.py').write_text(helper)
    cuda_helper = helper.replace('constraints="=f,f",', 'constraints="=r,f",').replace('dtype=tl.float32,\n            is_pure=True,', 'dtype=tl.uint32,\n            is_pure=True,')
    ast.parse(cuda_helper)
    (ROOT / 'cuda_precision.py').write_text(cuda_helper)
    for name in list(candidates):
        cuda_source = (ROOT / (name + '.py')).read_text().replace('from frozen_precision import (', 'from cuda_precision import (')
        path = ROOT / ('cuda_' + name + '.py')
        path.write_text(cuda_source)
        candidates['cuda_' + name] = {**candidates[name], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope': 'H100 compiler adaptation only: TF32 conversion uses a b32 destination register. Same Q-once algorithms.'}
    manifest = {'status': 'ISOLATED_CANDIDATES_GENERATED', 'candidates': candidates,
        'snapshot_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'frozen_snapshot').glob('*.py')},
        'helper_sha256': hashlib.sha256((ROOT / 'frozen_precision.py').read_bytes()).hexdigest(),
        'cuda_helper_sha256': hashlib.sha256((ROOT / 'cuda_precision.py').read_bytes()).hexdigest(),
        'limits': ['H100 candidates are custom FlagGems-derived kernels, not native CUE.',
                   'No production source or default server package modified.',
                   'RNE bit conversion must pass full bit-pattern checks before considering promotion.']}
    (ROOT / 'candidate_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('ISOLATED_Q_ONCE_CANDIDATES_GENERATED', list(candidates))


if __name__ == '__main__':
    main()
