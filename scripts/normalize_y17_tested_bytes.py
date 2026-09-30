"""Normalize only audited diagnostic copies that differ solely by CRLF."""

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import yaml


root = Path(__file__).resolve().parents[1]
head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
paths = subprocess.check_output([
    "git", "-C", str(root), "diff", "--name-only", "--diff-filter=A",
    "2bb1c107b27fbfdbdf2d6ffd25b43bb5bf4f3e6c", head, "--", "AdaRES"]).decode().splitlines()
rows = []
for name in paths:
    local = (root/name).read_bytes()
    blob = subprocess.check_output(["git", "-C", str(root), "show", head+":"+name])
    assert local.replace(b"\r\n", b"\n") == blob
    rows.append(dict(path=name[7:], before=hashlib.sha256(local).hexdigest(),
                     after=hashlib.sha256(blob).hexdigest(),
                     data=base64.b64encode(blob).decode()))
payload = base64.b64encode(json.dumps(rows).encode()).decode()
program = r'''
import base64,fcntl,hashlib,json,os,pathlib,sys,tempfile
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project/AdaRES')
lock=open(root/'.codex_train_entry_sync.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sha=lambda b:hashlib.sha256(b).hexdigest()
assert (root/'.codex_source_commit').read_text().strip()=='2bb1c107b27fbfdbdf2d6ffd25b43bb5bf4f3e6c'
result=[]; checked=[]
for r in json.loads(base64.b64decode(sys.argv[1]).decode('utf-8')):
    path=root/r['path']
    assert root.resolve() in path.resolve().parents
    if not path.exists(): continue
    old=path.read_bytes(); new=base64.b64decode(r['data'])
    assert sha(old) in (r['before'],r['after']), (r['path'],sha(old),r['before'],r['after'])
    assert old.replace(b'\r\n',b'\n')==new and sha(new)==r['after']
    checked.append((r,path,old,new))
for r,path,old,new in checked:
    if old!=new:
        fd,tmp=tempfile.mkstemp(dir=str(path.parent))
        with os.fdopen(fd,'wb') as f: f.write(new);f.flush();os.fsync(f.fileno())
        os.chmod(tmp,path.stat().st_mode)
        assert path.read_bytes()==old
        os.replace(tmp,path)
    assert sha(path.read_bytes())==r['after']
    result.append(dict(path=r['path'],verified=True,normalized=old!=new))
print(json.dumps(result))
'''
program = program.replace("sys.argv[1]", repr(payload))
server = yaml.safe_load((root/"SERVERS.yaml").read_text(encoding="utf-8"))["servers"]["server-2"]
assert server["remote_code_root"] == "/bak_home/songcl/Project/AdaRES_project/AdaRES"
ret = subprocess.run(["ssh", "-o", "BatchMode=yes", server["user"]+"@"+server["host"],
                      server["python"]+" -"], input=program.encode(),
                     capture_output=True, timeout=30)
if ret.returncode:
    print(ret.stderr.decode(errors="replace"))
    ret.check_returncode()
result = dict(head=head, source_marker_unchanged=True, rows=json.loads(ret.stdout))
(root/"pipeline/analyses/y17_byte_normalization_20260930.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
