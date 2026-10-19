"""Identical positive/perturbed reference controls on both evaluation stacks."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from biotite.structure.io.pdbx import CIFFile, CIFColumn

root = Path(os.environ['QUALITY_ROOT']); root.mkdir(parents=True, exist_ok=True)
reference = Path(os.environ['QUALITY_REFERENCE'])
pxm = Path(os.environ['QUALITY_PXM'])
target = root/'perturbed_reference.cif'
cif = CIFFile.read(reference); cat = cif.block['atom_site']
x = cat['Cartn_x'].as_array(float)
i = np.arange(len(x))
x = x+np.where(i%3 == 0, 2.0, 0.0)+np.where(i%7 == 0, -5.0, 0.0)
cat['Cartn_x'] = CIFColumn(np.asarray([f'{v:.3f}' for v in x]))
cif.write(target)
rows=[]
for kind, model in [('self',reference),('perturbed',target)]:
    output = root/(kind+'.json')
    p = subprocess.run([sys.executable,str(pxm),'-r',str(reference),'-m',str(model),'-o',str(output)],capture_output=True,text=True,timeout=300)
    (root/(kind+'.log')).write_text(p.stdout+p.stderr)
    d=json.loads(output.read_text()) if output.exists() else {}
    dockq = [v['dockq'] for v in d.get('interface',{}).values() if isinstance(v,dict) and isinstance(v.get('dockq'),(int,float))]
    rows.append(dict(kind=kind,returncode=p.returncode,lddt=d.get('complex',{}).get('lddt'),dockq=dockq))
summary=dict(reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
             perturbed_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),records=rows)
(root/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
