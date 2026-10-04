"""Persist selected source hashes and device/version scalars, without tensor files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--server',choices=('ppu','h100'),required=True)
p.add_argument('--work',type=Path,required=True)
p.add_argument('--baseline',type=Path)
args=p.parse_args()
extra=os.environ.get('PROTENIX_EXTRA_SITE_PACKAGES')
if extra:
    sys.path.append(extra)
import torch
import triton
sources={}
if args.server=='ppu':
    for label,package in (('baseline',args.baseline),('gemv',args.work/'column_candidate'),
                          ('bmm_candidate',args.work/'column_bmm_candidate')):
        for name in ('mm','addmm','bmm','rank_one','mv','mm_gemv'):
            path=package/'flag_gems/ops'/f'{name}.py'
            if path.exists():
                sources[label+'/'+name]=hashlib.sha256(path.read_bytes()).hexdigest()
result={'server':args.server,'torch':torch.__version__,'triton':triton.__version__,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device_name':torch.cuda.get_device_name(0),'sources':sources,
        'scripts':{x.name:hashlib.sha256(x.read_bytes()).hexdigest()
                   for x in args.work.iterdir() if x.suffix in ('.py','.sh')},
        'production_package_modified':False}
(args.work/'source_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print('SOURCE_INVENTORY_OK',args.server)
