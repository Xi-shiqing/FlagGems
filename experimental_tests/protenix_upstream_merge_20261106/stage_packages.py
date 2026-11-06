"""Chunk source archives over the existing documented persistent SSH transport."""
import base64
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SOCKET = Path('/tmp/protenix_ppu229_hostname_20261028.sock')
TRANSPORT = HERE.parent/'ppu_full_acceptance_20261005/precision_residual_20261019/persistent_documented_ppu229_hostname_20261028.py'
REMOTE = '/root/xishiqing_protenix_precision_20261023/upstream_merge_20261106'


def remote(command):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(260)
        client.connect(str(SOCKET))
        client.sendall((json.dumps({'command': command})+'\n').encode())
        response = json.loads(client.makefile('rb').readline())
    assert response['returncode'] == 0, response
    return response['stdout']


def python(code):
    return json.loads(remote("python3 - <<'SOURCE_PACKAGE'\n"+code+"\nSOURCE_PACKAGE\n"))


def ensure_session():
    if SOCKET.exists():
        assert remote('hostname\n').strip() == 'dsw-766246-9578bd86-vmm4s'
        return
    log = (HERE/'persistent_transport.log').open('ab')
    child = subprocess.Popen([sys.executable, str(TRANSPORT), 'ppu'], stdin=subprocess.PIPE,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    child.stdin.write(b'true\n'); child.stdin.close()
    (HERE/'persistent_transport.pid').write_text(str(child.pid))
    for _ in range(120):
        if SOCKET.exists():
            assert remote('hostname\n').strip() == 'dsw-766246-9578bd86-vmm4s'
            return
        assert child.poll() is None, 'Inspect persistent_transport.log before retry'
        time.sleep(.25)
    raise TimeoutError('Documented persistent connection did not become ready')


if __name__ == '__main__':
    ensure_session()
    receipts = {}
    for variant in sys.argv[1:] or ['ours', 'upstream']:
        local = HERE/(variant+'_runtime.tar.gz')
        data = local.read_bytes(); sha = hashlib.sha256(data).hexdigest()
        dest = REMOTE+'/'+local.name
        current = python('import pathlib,json\np=pathlib.Path('+repr(dest)+')\nprint(json.dumps(p.stat().st_size if p.exists() else 0))')
        assert current <= len(data)
        for start in range(current, len(data), 32768):
            chunk = base64.b64encode(data[start:start+32768]).decode()
            python('import pathlib,base64,json\np=pathlib.Path('+repr(dest)+')\nassert (p.stat().st_size if p.exists() else 0)=='+str(start)+'\nwith p.open("ab") as f:f.write(base64.b64decode('+repr(chunk)+'))\nprint(json.dumps(p.stat().st_size))')
            if start // 32768 % 16 == 0:
                print('SOURCE_UPLOAD', variant, start, len(data), flush=True)
        receipt = python('''import pathlib,tarfile,hashlib,json
archive=pathlib.Path(%r);target=archive.parent/'packages'/%r
assert hashlib.sha256(archive.read_bytes()).hexdigest()==%r
assert not target.exists(),'Do not overwrite an existing source package'
target.mkdir(parents=True)
with tarfile.open(archive) as source:
 for item in source:
  assert item.isfile() and item.name.startswith('flag_gems/') and '..' not in pathlib.PurePosixPath(item.name).parts
  path=target/item.name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(source.extractfile(item).read())
files={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file()}
receipt={'archive_sha256':%r,'root':str(target),'file_count':len(files),'tree_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
(target.parent/(%r+'_source_manifest.json')).write_text(json.dumps({'receipt':receipt,'files':files},indent=2)+'\\n')
print(json.dumps(receipt))
''' % (dest, variant, sha, sha, variant))
        receipts[variant] = receipt
        (HERE/'package_deployment.json').write_text(json.dumps(receipts,indent=2)+'\n')
        print('SOURCE_PACKAGE_READY', json.dumps(receipt), flush=True)
