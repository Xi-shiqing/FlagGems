"""Stage final merge changes as a new immutable complete source package."""
import base64
import hashlib
import json
from pathlib import Path

from manage_merge import call, REMOTE

HERE = Path(__file__).resolve().parent
REPO = Path('/home/lgx1429541889/upstream-merge-20261106/FlagGems')
files = {str(p.relative_to(REPO/'src')):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in (REPO/'src/flag_gems').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
previous = json.loads((HERE/'merged_v2_all_source_hashes.json').read_text())
changed = {name:base64.b64encode((REPO/'src'/name).read_bytes()).decode()
           for name, digest in files.items() if previous.get(name) != digest}
deleted = sorted(set(previous)-set(files))
tree = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
manifest = {'file_count':len(files),'tree_sha256':tree,'base':'merged_v2',
            'changed_hashes':{name:files[name] for name in changed},'deleted':deleted}
(HERE/'merged_final_package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(HERE/'merged_final_all_source_hashes.json').write_text(json.dumps(files,indent=2)+'\n')
code = '''import pathlib,json,base64,shutil,hashlib
root=pathlib.Path(%r);target=root/'packages/merged_final';assert not target.exists()
shutil.copytree(root/'packages/merged_v2',target)
for name,packed in %r.items():(target/name).write_bytes(base64.b64decode(packed))
for name in %r:(target/name).unlink()
files={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file()}
tree=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest();assert tree==%r,tree
r={'root':str(target),'file_count':len(files),'tree_sha256':tree};(root/'merged_final_deployment.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))
''' % (REMOTE, changed, deleted, tree)
result = call(code)
(HERE/'merged_final_deployment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
