"""Four-mode deterministic controls with exactly representable Q/K/V."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from audit_routes import metrics, tensor_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', choices=('ppu', 'h100'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.server == 'ppu':
        from flag_gems.ops import triangle_attention as entry
    else:
        from cuequivariance_ops_torch import triangle_attention as entry
    n, heads, d = 245, 4, 32
    ramp = torch.arange(n, device='cuda', dtype=torch.float32) / 256
    shape = (n, heads, n, d)
    q, k, v = [torch.zeros(shape, device='cuda') for _ in range(3)]
    k[..., 0] = ramp
    v[..., 0] = ramp
    bias = torch.zeros((1, heads, n, n), device='cuda')
    mask = torch.ones((n, 1, 1, n), device='cuda', dtype=torch.bool)
    rows = []
    for query, scale in ((32., 1.), (128., .125), (128., 32**-.5), (32., .25)):
        q[..., 0] = query
        scale32 = torch.tensor(scale, dtype=torch.float32).item()
        weights = torch.softmax((ramp.cpu().double() * query * scale32), -1)
        gold = torch.zeros(shape, dtype=torch.float64)
        gold[..., 0] = (weights * ramp.cpu().double()).sum()
        row = {'query_scalar': query, 'scale': scale, 'scale_fp32': scale32,
               'gold_qk_max_natural': query * scale32 * (n - 1) / 256,
               'canonical_inputs': [tensor_meta(value) for value in (q, k, v)], 'modes': {}}
        for mode in ('fp32', 'tf32'):
            torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
            torch.backends.cudnn.allow_tf32 = mode == 'tf32'
            def call():
                if args.server == 'h100':
                    return entry(q, k, v, bias, mask=mask, scale=scale, return_aux=True)
                inputs = [(q * scale).permute(0, 2, 1, 3).unsqueeze(0).contiguous(),
                          k.permute(0, 2, 1, 3).unsqueeze(0).contiguous(),
                          v.permute(0, 2, 1, 3).unsqueeze(0).contiguous()]
                return entry(*inputs, torch.zeros((1, n, 1, 1, n), device='cuda'),
                             bias.unsqueeze(0), precision='ieee' if mode == 'fp32' else 'tf32', layout='BNSHD')
            with torch.no_grad():
                result = call()
                if args.server == 'h100':
                    value, lse, maximum = result
                else:
                    value = result.squeeze(0).permute(0, 2, 1, 3)
                row['modes'][mode] = {'output': tensor_meta(value), 'vs_fp64_softmax_gold': metrics(value.cpu(), gold)}
                if args.server == 'h100':
                    row['modes'][mode]['maximum_mean'] = float(maximum.double().mean())
                    row['modes'][mode]['lse_mean'] = float(lse.double().mean())
                if query == 32. and scale == 1.:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                        call()
                        torch.cuda.synchronize()
                    row['modes'][mode]['kernel_trace'] = [event.name for event in prof.events()
                                                          if 'CUDA' in str(event.device_type)]
        if args.server == 'h100':
            row['tf32_vs_fp32_maximum_gain'] = row['modes']['tf32']['maximum_mean'] / row['modes']['fp32']['maximum_mean'] - 1
        rows.append(row)
    args.output.write_text(json.dumps({'status': 'PASS', 'server': args.server, 'records': rows,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Constructed exact binary inputs, no RNG calls, one full CPU FP64 analytic softmax gold. Accuracy and kernel identity only; no latency claim.'}, indent=2) + '\n')
    print('EXACT_TRIANGLE_INPUT_CONTROL_PASS', args.server)


if __name__ == '__main__':
    main()
