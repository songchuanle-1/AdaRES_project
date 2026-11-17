"""Read-only bounded launch evidence for the two authorized original runs."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess


root = Path(__file__).resolve().parents[1]
queue = json.loads((root / "pipeline/queue.json").read_text(encoding="utf-8"))
tasks = queue["gpu_queues"]["server-2"]
head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
paths = subprocess.check_output([
    "git", "-C", str(root), "diff", "--name-only",
    "42ffd589bfb5c27fa18d735866ddefa3a543e2ee", head, "--", "AdaRES"]).decode().splitlines()
files = []
for path in paths:
    blob = subprocess.check_output(["git", "-C", str(root), "show", head + ":" + path])
    files.append({"path": path[7:], "sha256": hashlib.sha256(blob).hexdigest()})
program = r'''
import fcntl,hashlib,json,pathlib,re,subprocess
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project/AdaRES')
d=PAYLOAD
files=[dict(path=r['path'],sha256=hashlib.sha256((root/r['path']).read_bytes()).hexdigest()) for r in d['files']]
assert files==d['files']
rows=[]
for task in d['tasks']:
    path=pathlib.Path(task['log'])
    with path.open('rb') as handle:
        handle.seek(max(0,path.stat().st_size-131072))
        text=handle.read().decode('utf-8',errors='replace')
    steps=[dict(epoch=int(e),total=int(t),step=int(s),batches=int(b))
           for e,t,s,b in re.findall(r'Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)',text)]
    unique=[]
    for step in steps:
        if not unique or step!=unique[-1]: unique.append(step)
    assert len(unique)>=2 and unique[-1]['step']>unique[-2]['step']
    cmd=pathlib.Path('/proc/'+str(task['pid'])+'/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert task['config'] in cmd and '--gpu_ids '+task['gpu'] in cmd
    rows.append(dict(id=task['id'],pid=task['pid'],gpu=task['gpu'],log=task['log'],
                     steps=unique[-3:],command=cmd,launch_verified=task['launch_verified'],
                     bounded_tail_has_traceback='Traceback (most recent call last)' in text))
lease=open(str(root.parent/'.y15_gpu1_diagnostic.lock'),'a')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
fcntl.flock(lease,fcntl.LOCK_UN)
lease.close()
print(json.dumps(dict(source_marker=(root/'.codex_source_commit').read_text().strip(),
                      remote_files=files,tasks=rows,diagnostic_lease_free=True,
                      hostname=subprocess.check_output(['hostname']).decode().strip(),
                      gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode())))
'''
program = program.replace("PAYLOAD", repr({"tasks": tasks, "files": files}))
response = subprocess.run([
    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    "songcl@192.168.1.163", "/home/songcl/.conda/envs/adares_project/bin/python -"],
    input=program.encode(), capture_output=True, check=True, timeout=60)
result = json.loads(response.stdout)
if result["source_marker"] != head or result["hostname"] != "dl01-MS-7C60":
    raise RuntimeError("Deployment marker/host identity mismatch")
result["observed_at_local"] = datetime.now().astimezone().isoformat()
result["candidate"] = "AdaRES-Y18"
result["scientific_status"] = "training verified; efficacy unknown"
path = root / "pipeline/analyses/y18_launch_verified_3efff5f3.json"
if path.exists():
    raise RuntimeError("Refusing overwrite launch evidence")
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "remote_files"}))
