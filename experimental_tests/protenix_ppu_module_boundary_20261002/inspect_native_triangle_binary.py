"""Read installed CUE embedded kernels; persist only instruction/hash summaries.

Extracted binary/disassembly stays on its original server. No installed file is
modified. Register names are an audit of this binary, not a portable API.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    data = args.library.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    kernel = 'native_sm80_fprop_fmha_tf32_64x32x32'
    records = []
    for index, match in enumerate(re.finditer(b'\x7fELF', data)):
        start = match.start()
        if data[start + 4:start + 6] != bytes((2, 1)):
            continue
        if struct.unpack_from('<H', data, start + 18)[0] != 190:
            continue
        flags = struct.unpack_from('<I', data, start + 48)[0]
        sm = flags & 255
        if sm not in (80, 90):
            continue
        shoff = struct.unpack_from('<Q', data, start + 40)[0]
        shsize, shnum = struct.unpack_from('<HH', data, start + 58)
        if shsize != 64 or shnum > 10000 or start + shoff + shsize * shnum > len(data):
            continue
        end = shoff + shsize * shnum
        for i in range(shnum):
            at = start + shoff + i * shsize
            kind = struct.unpack_from('<I', data, at + 4)[0]
            offset, size = struct.unpack_from('<QQ', data, at + 24)
            if kind != 8:  # SHT_NOBITS has no file bytes.
                end = max(end, offset + size)
        phoff = struct.unpack_from('<Q', data, start + 32)[0]
        phsize, phnum = struct.unpack_from('<HH', data, start + 54)
        end = max(end, phoff + phsize * phnum)
        for i in range(phnum):
            at = start + phoff + i * phsize
            offset = struct.unpack_from('<Q', data, at + 8)[0]
            size = struct.unpack_from('<Q', data, at + 32)[0]
            end = max(end, offset + size)
        binary = data[start:start + end]
        if kernel.encode() not in binary:
            continue
        path = args.work / f'installed_triangle_{index}.cubin'
        path.write_bytes(binary)
        listing = subprocess.check_output(['/usr/local/cuda/bin/nvdisasm', str(path)], text=True)
        (args.work / f'installed_triangle_{index}.disasm').write_text(listing)
        marker = '//--------------------- .text.' + kernel
        first = listing.index(marker)
        last = listing.find('\n//---------------------', first + len(marker))
        block = listing[first:last if last >= 0 else None]
        instructions = []
        for line in block.splitlines():
            item = re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?P\d+\s+)?(\S+)', line)
            if item:
                instructions.append((item[1], item[2], line))
        records.append({'sm': sm, 'kernel': kernel, 'library_offset': start, 'binary_bytes': end,
            'binary_sha256': hashlib.sha256(binary).hexdigest(),
            'instruction_counts': dict(Counter(op for _, op, _ in instructions)),
            'register_32_read_write_sites': [{'offset': at, 'opcode': op,
                'self_integer_add': bool(re.search(r'IMAD.IADD.U32 R32, R32,', line))}
                for at, op, line in instructions if re.search(r'\sR32,', line)],
            'rounding_bias_0x1000_present': '0x1000' in block})
    assert records and hashlib.sha256(args.library.read_bytes()).hexdigest() == digest
    (args.work / 'native_binary_audit.json').write_text(json.dumps({
        'status': 'READ_ONLY_AUDIT_PASS', 'library_sha256': digest, 'records': records,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Installed SM80/SM90 code identities and instruction counts; does not independently prove driver binary selection or numerical causality.'}, indent=2) + '\n')
    print('NATIVE_TRIANGLE_BINARY_AUDIT_PASS', [(r['sm'], r['binary_bytes']) for r in records])


if __name__ == '__main__':
    main()
