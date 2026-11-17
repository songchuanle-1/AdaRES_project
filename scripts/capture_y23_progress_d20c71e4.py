"""Bounded source and original training-progress evidence under the owned tick."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
TOKEN="d20c71e418354a6b818d9f37ab72fc7b"
assert json.loads((ROOT/"pipeline/headless/.foreground.lockdir/owner.json").read_text())["token"]==TOKEN
queue=json.loads((ROOT/"pipeline/queue.json").read_text(encoding="utf-8"))
tasks=[t for t in queue["gpu_queues"]["server-2"] if t["id"] in ("AdaRES-Y18","AdaRES-Y23")]
assert len(tasks)==2 and all(t["status"]=="running" and t["launch_verified"] for t in tasks)
files=json.loads((ROOT/"pipeline/analyses/y23_source_bundle_d20c71e4.json").read_text())["files"]
head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"]).decode().strip()
program=r'''
import fcntl,hashlib,json,pathlib,re,subprocess,time
payload=__PAYLOAD__
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project/AdaRES')
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
assert (root/'.codex_source_commit').read_text().strip()==payload['head']
files=[dict(path=r['path'],sha256=hashlib.sha256((root.parent/r['path']).read_bytes()).hexdigest()) for r in payload['files']]
assert files==payload['files']
lease=open(str(root.parent/'.y15_gpu0_diagnostic.lock'),'a')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
rows=[]
for task in payload['tasks']:
    cmd=pathlib.Path('/proc/'+str(task['pid'])+'/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert task['config'] in cmd and '--gpu_ids '+task['gpu'] in cmd
    path=pathlib.Path(task['log']);captures=[]
    for obs in range(2):
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-24000));raw=f.read()
        text=raw.decode('utf-8',errors='replace')
        steps=re.findall(r'Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)',text)
        assert steps
        epoch,total,step,batches=map(int,steps[-1])
        captures.append(dict(epoch=epoch,total=total,step=step,batches=batches,
            tail1200_sha256=hashlib.sha256(raw[-1200:]).hexdigest(),
            bounded_traceback='Traceback (most recent call last)' in text,
            bounded_oom='CUDA out of memory' in text))
        if not obs:time.sleep(3)
    assert (captures[1]['epoch'],captures[1]['step'])>(captures[0]['epoch'],captures[0]['step'])
    rows.append(dict(id=task['id'],pid=task['pid'],gpu=task['gpu'],log=task['log'],
        captures=captures,launch_verified=True))
print(json.dumps(dict(source_marker=payload['head'],remote_files=files,tasks=rows,
    diagnostic_flock_released=True,
    gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode())))
'''.replace("__PAYLOAD__",repr(dict(tasks=tasks,files=files,head=head)))
start=datetime.now().astimezone().isoformat()
r=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15","songcl@192.168.1.163",
    "/home/songcl/.conda/envs/adares_project/bin/python -"],input=program.encode(),
    capture_output=True,timeout=60)
out=ROOT/"pipeline/analyses/y23_y18_progress_d20c71e4.json"
log=out.with_suffix(".log")
assert not out.exists() and not log.exists()
log.write_bytes(r.stdout+b"\nSTDERR:\n"+r.stderr)
if r.returncode:
    raise RuntimeError("Progress capture failed; retained "+str(log))
result=json.loads(r.stdout)
result.update(observed_start_local=start,observed_end_local=datetime.now().astimezone().isoformat(),
    scientific_status="Y23/Y18 training verified, final efficacy unknown",
    scope="Two bounded native observations; absence of errors only within read tails, not full-log proof")
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k!="remote_files"},indent=2))
