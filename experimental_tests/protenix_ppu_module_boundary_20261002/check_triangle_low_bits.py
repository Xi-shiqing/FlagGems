"""Counterfactual inputs cancel native integer rounding bias; no production change."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from audit_routes import metrics, tensor_meta


def cancel_integer_bias(value):
    bits = value.contiguous().view(torch.int32)
    # Preserve exact zeros and finite values; target values in this test are TF32 exact.
    return torch.where(value != 0, bits - 0x1000, bits).view(torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from cuequivariance_ops_torch import triangle_attention
    n, heads, d = 245, 4, 32
    shape = (n, heads, n, d)
    ramp = torch.arange(n, device='cuda', dtype=torch.float32) / 256
    q, k, v = [torch.zeros(shape, device='cuda') for _ in range(3)]
    k[..., 0] = ramp
    v[..., 0] = ramp
    bias = torch.zeros((1, heads, n, n), device='cuda')
    mask = torch.ones((n, 1, 1, n), device='cuda', dtype=torch.bool)
    rows = []
    with torch.no_grad():
        for query, scale in ((32., 1.), (1.5, 1.), (128., 32**-.5), (-32., 1.)):
            q[..., 0] = query
            target = (q, k, v)
            changed = tuple(cancel_integer_bias(x) for x in target)
            scale32 = torch.tensor(scale, dtype=torch.float32).item()
            weights = torch.softmax(ramp.cpu().double() * query * scale32, -1)
            gold = torch.zeros((1, *shape), dtype=torch.float64)
            gold[..., 0] = (weights * ramp.cpu().double()).sum()
            row = {'query': query, 'scale': scale, 'canonical_inputs': [tensor_meta(x) for x in target], 'probes': {}}
            for name, indexes in (('fp32', ()), ('tf32', ()), ('q_cancel', (0,)), ('k_cancel', (1,)),
                                  ('v_cancel', (2,)), ('qk_cancel', (0, 1)), ('qkv_cancel', (0, 1, 2))):
                torch.backends.cuda.matmul.allow_tf32 = name != 'fp32'
                torch.backends.cudnn.allow_tf32 = name != 'fp32'
                values = tuple(changed[i] if i in indexes else x for i, x in enumerate(target))
                output, lse, maximum = triangle_attention(*values, bias, mask, scale, return_aux=True)
                row['probes'][name] = {'vs_original_input_fp64_gold': metrics(output.cpu(), gold),
                    'maximum_mean': float(maximum.double().mean()), 'lse_mean': float(lse.double().mean())}
            rows.append(row)
    args.output.write_text(json.dumps({'status': 'EXECUTION_PASS', 'records': rows,
        'scope': 'Counterfactual input manipulation to test native bit preparation; errors reference ORIGINAL inputs, not accuracy of manipulated mathematical inputs. No model or performance claim.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
    print('TRIANGLE_LOW_BITS_DONE', len(rows))


if __name__ == '__main__':
    main()
