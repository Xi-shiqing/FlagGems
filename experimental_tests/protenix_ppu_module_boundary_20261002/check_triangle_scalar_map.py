"""Native fused triangle score controls: exact scalars, no model tensor export."""
import hashlib
import json
from pathlib import Path
import argparse
import torch
from audit_routes import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from cuequivariance_ops_torch import triangle_attention
    shape = (245, 4, 245, 32)
    q, k = [torch.zeros(shape, device='cuda') for _ in range(2)]
    v = torch.full(shape, .25, device='cuda')
    bias = torch.zeros((1, 4, 245, 245), device='cuda')
    mask = torch.ones((245, 1, 1, 245), device='cuda', dtype=torch.bool)
    cases = [(x, 1., 1, 1.) for x in (0., 1., 1.125, 1.5, 1.75, 2., 4., 32., -1., -1.5)]
    cases += [(1., x, 1, 1.) for x in (1.125, 1.5, 1.75, -1., -1.5)]
    cases += [(1., 1., c, 1.) for c in (2, 8, 32)]
    cases += [(1., 1., 1, s) for s in (.125, .25, 32**-.5)]
    rows = []
    with torch.no_grad():
        for a, b, channels, scale in cases:
            q.zero_(); k.zero_()
            q[..., :channels] = a
            k[..., :channels] = b
            gold = a * b * channels * torch.tensor(scale, dtype=torch.float32).item()
            row = {'q': a, 'k': b, 'active_channels': channels, 'scale': scale,
                   'gold_score': gold, 'modes': {}}
            for mode in ('fp32', 'tf32'):
                torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
                torch.backends.cudnn.allow_tf32 = mode == 'tf32'
                output, lse, maximum = triangle_attention(q, k, v, bias, mask, scale, return_aux=True)
                row['modes'][mode] = {'maximum_mean': float(maximum.double().mean()),
                    'maximum_min': float(maximum.min()), 'maximum_max': float(maximum.max()),
                    'lse_mean': float(lse.double().mean()),
                    'output_vs_constant_gold': metrics(output.cpu(), torch.full(output.shape, .25))}
            rows.append(row)
    args.output.write_text(json.dumps({'status': 'EXECUTION_PASS', 'server': 'h100',
        'scope': 'Synthetic exact scalars; native fused score/auxiliary diagnostics only, no latency or model accuracy claim.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'records': rows}, indent=2) + '\n')
    print('TRIANGLE_SCALAR_MAP_DONE', len(rows))


if __name__ == '__main__':
    main()
