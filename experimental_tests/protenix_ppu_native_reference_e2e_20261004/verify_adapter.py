"""Check scale, mask, batch and layout mapping against CPU FP64."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from triangle_adapter import triangle_attention
from cuequivariance_ops_torch import triangle_attention as native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for index, shape in enumerate(((1, 3, 2, 67, 16), (2, 5, 8, 129, 32))):
        b, n, h, s, d = shape
        rng = np.random.Generator(np.random.PCG64(101 + index))
        arrays = [rng.integers(-32768, 32768, size=shape, dtype=np.int32).astype(np.float32) * 2**-16 for _ in range(3)]
        bias = rng.integers(-8192, 8192, size=(b, 1, h, s, s), dtype=np.int32).astype(np.float32) * 2**-16
        mask = rng.integers(0, 3, size=(b, n, 1, 1, s)) != 0
        mask[..., 0] = True
        for scale in (None, .125, 1.):
            q, k, v = [torch.from_numpy(a).double() for a in arrays]
            gold = ((q * (d**-.5 if scale is None else scale)) @ k.transpose(-1, -2) + torch.from_numpy(bias).double())
            gold = (gold + torch.from_numpy((~mask).astype(np.float64) * -1e9)).softmax(-1) @ v
            for squeezed in (False, True) if b == 1 else (False,):
                values = [torch.from_numpy(a).cuda() for a in arrays + [bias, mask]]
                if squeezed:
                    values = [a.squeeze(0) for a in values]
                result = triangle_attention(*values, scale=scale)
                reference = native(*values, scale=scale)
                assert result.shape == reference.shape == gold.shape, (squeezed, result.shape, reference.shape)
                out = result.detach().cpu().double()
                error = float((out - gold).norm() / gold.norm())
                assert error < 5e-4 and torch.isfinite(out).all(), (shape, scale, error)
                rows.append({'shape': shape, 'scale': scale, 'squeezed': squeezed, 'relative_l2_vs_cpu_fp64': error,
                    'input_sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]})
    args.output.write_text(json.dumps({'status': 'PASS', 'cases': rows,
        'scope': 'Adapter scale/mask/layout check; full native correctness is not inferred from this test.'}, indent=2)+'\n')
    print('CORRECTED_TRIANGLE_ADAPTER_PASS', len(rows))


if __name__ == '__main__':
    main()
