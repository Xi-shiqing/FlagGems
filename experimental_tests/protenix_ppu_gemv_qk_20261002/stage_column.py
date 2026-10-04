"""Keep measured row-vector candidates intact; stage column-result MM only."""
import hashlib
import json
from pathlib import Path
import shutil

root=Path(__file__).resolve().parent
sources={}
for old,new in (('candidate','column_candidate'),('bmm_candidate','column_bmm_candidate')):
    dest=root/new
    assert not dest.exists()
    shutil.copytree(root/old,dest,ignore=shutil.ignore_patterns('__pycache__','.git'))
    ops=dest/'flag_gems/ops'
    shutil.copy2(root/'mm_gemv_column.py',ops/'mm_gemv.py')
    path=ops/'mm.py'
    text=path.read_text()
    assert text.count('(M == 1 or N == 1)')==2
    path.write_text(text.replace('(M == 1 or N == 1)','N == 1'))
    compile(path.read_text(),str(path),'exec')
    sources[new]={name:hashlib.sha256((ops/(name+'.py')).read_bytes()).hexdigest()
                  for name in ('mm','mm_gemv','bmm')}
(root/'column_candidate_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
print('COLUMN_CANDIDATES_STAGED')
