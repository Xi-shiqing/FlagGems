"""Deduplicate existing scalar boundary JSON; never read model tensors."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    pool = {}
    modes = {}
    for mode, rows in data['modes'].items():
        modes[mode] = []
        for row in rows:
            compact = {k: v for k, v in row.items() if k not in ('inputs', 'outputs')}
            for side in ('inputs', 'outputs'):
                compact[side] = {}
                for path, value in row[side].items():
                    key = value['sha256'] + ':' + ','.join(map(str, value['shape']))
                    shared = {k: v for k, v in value.items() if k != 'vs_own_fp32'}
                    if key in pool:
                        assert pool[key] == shared
                    else:
                        pool[key] = shared
                    compact[side][path] = {'pool': key}
                    if 'vs_own_fp32' in value:
                        compact[side][path]['vs_own_fp32'] = value['vs_own_fp32']
            modes[mode].append(compact)
    result = {k: v for k, v in data.items() if k != 'modes'}
    result.update(modes=modes, pool=pool,
                  original_json_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print('COMPACT_BOUNDARY_OK', len(pool), args.output.stat().st_size)


if __name__ == '__main__':
    main()
