"""One valid key exposes score/LSE dependence on KV position without exporting tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from audit_routes import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from cuequivariance_ops_torch import triangle_attention
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
                output, lse, maximum = triangle_attention(q, k, v, bias, mask, 1., return_aux=True)
                row['modes'][mode] = {'maximum_mean': float(maximum.double().mean()),
                    'lse_mean': float(lse.double().mean()),
                    'output_vs_gold': metrics(output.cpu(), torch.full(output.shape, .25))}
            records.append(row)
    args.output.write_text(json.dumps({'status': 'EXECUTION_PASS', 'records': records,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'One identical valid key at varied KV positions, exact scalar score and LSE gold. Native diagnostic only; no model or timing claim.'}, indent=2) + '\n')
    print('TRIANGLE_KEY_POSITIONS_DONE', len(records))


if __name__ == '__main__':
    main()
