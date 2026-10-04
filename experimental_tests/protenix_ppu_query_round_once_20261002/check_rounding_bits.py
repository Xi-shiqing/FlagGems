"""Bit-pattern checks only; no operand tensors are persisted."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.pop('PROTENIX_TF32_CAPTURE_DIR', None)
os.environ['PROTENIX_TF32_RESTORE_MODULES'] = ''

import numpy as np
import torch
import triton
import triton.language as tl
if torch.cuda.get_device_name().startswith('NVIDIA'):
    from cuda_precision import round_tf32_rne, round_tf32_rne_bits, round_tf32_rna_bits
else:
    from frozen_precision import round_tf32_rne, round_tf32_rne_bits, round_tf32_rna_bits


@triton.jit
def convert(src, dst, size: tl.constexpr, mode: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + index, index < size, other=0).to(tl.float32, bitcast=True)
    if mode == 0:
        result = round_tf32_rne(value, True)
    elif mode == 1:
        result = round_tf32_rne_bits(value, True)
    else:
        result = round_tf32_rna_bits(value, True)
    tl.store(dst + index, result.to(tl.int32, bitcast=True), index < size)


def sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.Generator(np.random.PCG64(101))
    random = rng.integers(0, 2**32, 65536, dtype=np.uint32)
    # Both signs, all exponents, exact values, below/tie/above and overflow boundaries.
    edges = np.array([sign | exponent << 23 | mantissa
        for sign in (0, 0x80000000) for exponent in range(256)
        for mantissa in (0, 1, 0xfff, 0x1000, 0x1001, 0x1fff, 0x2000,
                         0x2fff, 0x3000, 0x3001, 0x7fefff, 0x7ff000, 0x7fffff)], dtype=np.uint32)
    bits = np.concatenate((random, edges))
    finite = (bits & np.uint32(0x7f800000)) != np.uint32(0x7f800000)
    classes = {'normal': finite & ((bits & 0x7f800000) != 0),
               'subnormal_or_zero': (bits & 0x7f800000) == 0,
               'inf_or_nan': ~finite}
    rne = np.where(finite, (bits + np.uint32(0xfff) + ((bits >> 13) & 1)) & np.uint32(0xffffe000), bits).astype(np.uint32)
    rna = np.where(finite, (bits + np.uint32(0x1000)) & np.uint32(0xffffe000), bits).astype(np.uint32)
    src = torch.from_numpy(bits.view(np.int32)).cuda()
    dst = torch.empty_like(src)
    rows = {}
    for mode, name, reference in ((0, 'hardware_rne', rne), (1, 'bit_rne', rne), (2, 'bit_rna', rna)):
        convert[(triton.cdiv(len(bits), 256),)](src, dst, len(bits), mode, 256)
        actual = dst.cpu().numpy().view(np.uint32).copy()
        mismatch = actual != reference
        converted_src = dst.clone()
        convert[(triton.cdiv(len(bits), 256),)](converted_src, dst, len(bits), mode, 256)
        twice = dst.cpu().numpy().view(np.uint32)
        rows[name] = {'sha256': sha(actual), 'mismatches': int(mismatch.sum()),
            'mismatches_by_class': {key: int((mismatch & mask).sum()) for key, mask in classes.items()},
            'idempotence_mismatches': int((twice != actual).sum()),
            'examples': [{'input_hex': hex(int(bits[i])), 'actual_hex': hex(int(actual[i])), 'expected_hex': hex(int(reference[i]))}
                         for i in np.flatnonzero(mismatch)[:8]]}
    result = {'status': 'PASS' if all(row['mismatches'] == row['idempotence_mismatches'] == 0 for row in rows.values()) else 'MISMATCH',
        'server': args.server, 'device': torch.cuda.get_device_name(), 'count': len(bits),
        'input_sha256': sha(bits), 'seed': 101, 'rng': 'NumPy PCG64 uint32; no GPU RNG',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'modes': rows,
        'scope': 'Full exponent coverage and 65536 random bit patterns. Scalars and hashes only; no persisted operands.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('ROUNDING_BITS_DONE', args.server, result['status'], {key: row['mismatches'] for key, row in rows.items()}, flush=True)


if __name__ == '__main__':
    main()
