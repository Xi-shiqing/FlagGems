"""One valid key exposes score/LSE dependence on KV position without exporting tensors."""
import argparse
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import torch
from audit_routes import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', choices=('ppu', 'h100'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.server == 'h100':
        from cuequivariance_ops_torch import triangle_attention
        impl = None
    else:
        from flag_gems.ops import triangle_attention
        impl = importlib.import_module('flag_gems.ops.triangle_attention')._streaming_triangle_attention_forward_impl
        source = inspect.getsourcefile(impl)
        code = Path(source).read_text()
        assert 'm_i + tl.math.log2(safe_l)' in code, 'verify private cache units from actual installed code'
    shape = (245, 4, 245, 32)
    q, k = [torch.zeros(shape, device='cuda') for _ in range(2)]
    q[..., 0] = 1.
    k[..., 0] = 1.
    v = torch.full(shape, .25, device='cuda')
    bias = torch.zeros((1, 4, 245, 245), device='cuda')
    mask = torch.zeros((245, 1, 1, 245), device='cuda', dtype=torch.bool)
    records = []
    with torch.no_grad():
        for position in (0, 1, 15, 16, 31, 32, 63, 64, 95, 96, 127, 128, 191, 192, 224, 244):
            mask.zero_()
            mask[..., position] = True
            row = {'key_position': position, 'gold_score': 1., 'gold_lse': 1., 'modes': {}}
            for mode in ('fp32', 'tf32'):
                torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
                torch.backends.cudnn.allow_tf32 = mode == 'tf32'
                if args.server == 'h100':
                    output, lse, maximum = triangle_attention(q, k, v, bias, mask, 1., return_aux=True)
                    extra = {'maximum_mean': float(maximum.double().mean())}
                else:
                    inputs = [x.permute(0, 2, 1, 3).unsqueeze(0).contiguous() for x in (q, k, v)]
                    bias1 = (-1e9 * (~mask).float()).permute(0, 2, 1, 3).unsqueeze(0)
                    precision = 'ieee' if mode == 'fp32' else 'tf32'
                    output = triangle_attention(*inputs, bias1, bias.unsqueeze(0), precision=precision, layout='BNSHD')
                    auxiliary_output, lse = impl(*inputs, bias1, bias.unsqueeze(0), precision=precision, save_lse=True)
                    assert metrics(output.cpu(), auxiliary_output.cpu())['max_abs'] == 0.
                    lse = lse.double() * math.log(2.)
                    extra = {'cache_units': 'Converted base-2 cache to natural log; actual installed source checked',
                        'entry_source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest()}
                row['modes'][mode] = {'lse_mean': float(lse.double().mean()),
                    'output_vs_gold': metrics(output.cpu(), torch.full(output.shape, .25)), **extra}
            records.append(row)
    args.output.write_text(json.dumps({'status': 'EXECUTION_PASS', 'server': args.server, 'records': records,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'One identical valid key at varied KV positions, exact scalar score and LSE gold. PPU public entry matches save-auxiliary entry. Operator diagnostic only; no model or timing claim.'}, indent=2) + '\n')
    print('TRIANGLE_KEY_POSITIONS_DONE', len(records))


if __name__ == '__main__':
    main()
