"""Read-only remote evidence of deployment hashes and increasing native steps."""

import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone


root = Path(__file__).resolve().parents[1]
queue = json.loads((root/"pipeline/queue.json").read_text(encoding="utf-8"))
tasks = queue["gpu_queues"]["server-2"]
head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
paths = subprocess.check_output([
    "git", "-C", str(root), "diff", "--name-only",
    "2bb1c107b27fbfdbdf2d6ffd25b43bb5bf4f3e6c", head, "--", "AdaRES"]).decode().splitlines()
files = []
for path in paths:
    blob = subprocess.check_output(["git", "-C", str(root), "show", head+":"+path])
    files.append(dict(path=path[7:], sha256=hashlib.sha256(blob).hexdigest()))
program = r'''
import hashlib,json,os,pathlib,re,subprocess,sys
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project/AdaRES')
d=json.loads(sys.argv[1])
files=[dict(path=r['path'],sha256=hashlib.sha256((root/r['path']).read_bytes()).hexdigest()) for r in d['files']]
assert files==d['files']
rows=[]
for task in d['tasks']:
    path=pathlib.Path(task['log'])
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-131072))
        text=f.read().decode('utf-8',errors='replace')
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
                     tail_has_traceback='Traceback (most recent call last)' in text))
print(json.dumps(dict(source_marker=(root/'.codex_source_commit').read_text().strip(),
                      remote_files=files,tasks=rows,
                      gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode())))
'''
program = program.replace("sys.argv[1]", repr(json.dumps(dict(tasks=tasks, files=files))))
output = subprocess.run(["ssh", "-o", "BatchMode=yes", "songcl@192.168.1.163",
                         "/home/songcl/.conda/envs/adares_project/bin/python -"],
                        input=program.encode(), capture_output=True, check=True, timeout=40)
result = json.loads(output.stdout)
assert result["source_marker"] == head
result["observed_at_local"] = datetime.now().astimezone().isoformat()
result["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
result["candidate"] = "AdaRES-Y17"
result["science"] = "native launch verified; trained efficacy unknown"
(root/"pipeline/analyses/y17_launch_verified_20260930.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "remote_files"}))
