"""Stage the final source delta in bounded chunks and verify the complete tree."""
import base64
import hashlib
import json
from pathlib import Path
import tarfile

from stage_packages import ensure_session, python, REMOTE

HERE = Path(__file__).resolve().parent
REPO = Path('/home/lgx1429541889/upstream-merge-20261106/FlagGems')
files = {str(p.relative_to(REPO/'src')):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in (REPO/'src/flag_gems').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
previous = json.loads((HERE/'merged_adopted_all_source_hashes.json').read_text())
changed = [name for name,digest in files.items() if previous.get(name) != digest]
deleted = sorted(set(previous)-set(files))
tree = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
archive = HERE/'restored_delta.tar.gz'
with tarfile.open(archive,'w:gz') as target:
    for name in changed:
        target.add(REPO/'src'/name,arcname=name,recursive=False)
manifest = {'base':'merged_adopted','file_count':len(files),'tree_sha256':tree,
            'changed_hashes':{name:files[name] for name in changed},'deleted':deleted,
            'delta_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
(HERE/'merged_restored_package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(HERE/'merged_restored_all_source_hashes.json').write_text(json.dumps(files,indent=2)+'\n')
ensure_session()
data=archive.read_bytes();remote_file=REMOTE+'/restored_delta.tar.gz'
for start in range(0,len(data),32768):
    chunk=base64.b64encode(data[start:start+32768]).decode()
    python('import pathlib,base64,json\np=pathlib.Path('+repr(remote_file)+')\nassert (p.stat().st_size if p.exists() else 0)=='+str(start)+'\nwith p.open("ab") as f:f.write(base64.b64decode('+repr(chunk)+'))\nprint(json.dumps(p.stat().st_size))')
code='''import pathlib,shutil,tarfile,json,hashlib
root=pathlib.Path(%r);target=root/'packages/merged_restored';assert not target.exists()
archive=root/'restored_delta.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==%r
shutil.copytree(root/'packages/merged_adopted',target)
with tarfile.open(archive) as source:
 for item in source:
  assert item.isfile() and item.name.startswith('flag_gems/') and '..' not in pathlib.PurePosixPath(item.name).parts
  path=target/item.name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(source.extractfile(item).read())
for name in %r:(target/name).unlink()
files={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file()}
tree=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest();assert tree==%r,tree
r={'root':str(target),'file_count':len(files),'tree_sha256':tree};(root/'merged_restored_deployment.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))
''' % (REMOTE,manifest['delta_sha256'],deleted,tree)
result=python(code)
(HERE/'merged_restored_deployment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
