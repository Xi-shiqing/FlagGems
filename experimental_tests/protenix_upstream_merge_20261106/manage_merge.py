"""Reuse original documented transport; upload code and retrieve scalar records only."""
import base64
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mm_accumulation_20261105'))
from manage import call

REMOTE = '/root/xishiqing_protenix_precision_20261023/upstream_merge_20261106'
OLD = '/mnt/cpfs/xishiqing/ppu_full_acceptance_20261005/precision_residual_20261019'


def upload(paths):
    contents = {str(p.relative_to(HERE)): base64.b64encode(p.read_bytes()).decode() for p in paths}
    code = '''import pathlib,base64,hashlib,json
root=pathlib.Path(%r);root.mkdir(exist_ok=True)
receipt={}
for name,packed in %r.items():
 p=root/name;p.parent.mkdir(exist_ok=True,parents=True);data=base64.b64decode(packed)
 if p.exists():assert p.read_bytes()==data,'Executed source must stay immutable: '+name
 else:p.write_bytes(data)
 receipt[name]=hashlib.sha256(data).hexdigest()
print(json.dumps(receipt))
''' % (REMOTE, contents)
    return call(code)


def launch(script, key):
    return call('''import pathlib,subprocess,json
root=pathlib.Path(%r)
assert not (root/(%r+'.pid')).exists(),'Already launched'
program="import pathlib,subprocess;root=pathlib.Path("+repr(str(root))+");log=(root/"+repr(%r+'.log')+").open('xb');rc=subprocess.call(['python3',str(root/"+repr(%r)+")],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);log.close();(root/"+repr(%r+'.exit')+").write_text(str(rc))"
child=subprocess.Popen(['python3','-S','-c',program],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
(root/(%r+'.pid')).write_text(str(child.pid))
print(json.dumps({'pid':child.pid,'root':str(root),'job':%r}))
''' % (REMOTE, key, key, script, key, key, key))


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'deploy':
        files = [HERE/'compare_conflicts.py', HERE/'queue_conflicts.py', HERE/'variant_source_hashes.json']
        files += sorted((HERE/'variants').rglob('*.py'))
        result = upload(files)
        (HERE/'deployment.json').write_text(json.dumps(result, indent=2)+'\n')
        result = launch('queue_conflicts.py', 'conflicts')
    elif action == 'status':
        result = call('''import pathlib,json
r=pathlib.Path(%r);p=r/'conflict_results';answer={}
for n in ['conflicts.exit','conflicts.pid','conflicts.log']:
 if (r/n).exists():answer[n]=(r/n).read_text()[-3500:]
if (p/'result.json').exists():
 j=json.loads((p/'result.json').read_text());answer['completed_cases']=len(j['cases']);answer['status']=j.get('status');answer['failed']=[{'name':x['name'],'variants':{v:z.get('exception') for v,z in x['variants'].items() if z['status']!='PASS'}} for x in j['cases'] if any(z['status']!='PASS' for z in x['variants'].values())]
print(json.dumps(answer))
''' % REMOTE)
    elif action == 'collect':
        result = call('''import pathlib,json
r=pathlib.Path(%r);print((r/'conflict_results/result.json').read_text())
''' % REMOTE)
        (HERE/'conflict_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
        result = {'status': result.get('status'), 'cases': len(result['cases'])}
    else:
        raise ValueError(action)
    print(json.dumps(result, ensure_ascii=False))
