"""Minimal synthetic CUE reproducer; no model, checkpoint or exported tensors."""
import argparse
import json
from importlib import metadata
from pathlib import Path
import torch
from cuequivariance_ops_torch import triangle_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    q = torch.zeros((1, 4, 245, 32), device='cuda')
    k = torch.zeros_like(q)
    q[..., 0] = k[..., 0] = 1.
    v = torch.full_like(q, .25)
    bias = torch.zeros((1, 4, 245, 245), device='cuda')
    mask = torch.zeros((1, 1, 1, 245), device='cuda', dtype=torch.bool)
    records = []
    with torch.no_grad():
        for position in (0, 1, 15, 16, 31, 32, 63, 64, 95, 96, 127, 128, 191, 192, 224, 244):
            mask.zero_()
            mask[..., position] = True
            row = {'key_position': position, 'gold_score_and_lse': 1.}
            for mode in ('fp32', 'tf32'):
                torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
                out, lse, maximum = triangle_attention(q, k, v, bias, mask, 1., return_aux=True)
                row[mode] = {'lse_mean': float(lse.double().mean()), 'maximum_mean': float(maximum.double().mean()),
                    'output_max_abs_error': float((out-.25).abs().max())}
            records.append(row)
    assert all(abs(r['fp32']['lse_mean']-1.) < 1e-6 for r in records)
    result = {'status': 'EXECUTION_PASS', 'torch': torch.__version__,
        'native_ops_version': metadata.version('cuequivariance-ops-torch-cu12'), 'records': records,
        'scope': 'Exact TF32-representable Q=K=1, one valid key, score=1; output stays constant so output-only checks miss the score/LSE defect.'}
    text = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
