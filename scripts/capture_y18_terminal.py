"""Capture bounded Y18 completion evidence without changing its lifecycle."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner_path = ROOT / "pipeline/headless/.foreground.lockdir/owner.json"


def check_lease():
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    assert owner["token"] == args.token
    assert owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])


check_lease()
server = yaml.safe_load((ROOT / "SERVERS.yaml").read_text(encoding="utf-8"))["servers"]["server-2"]
assert server["host"] == "192.168.1.163" and server["user"] == "songcl"
assert server["allowed_gpus"] == [1]
queue = json.loads((ROOT / "pipeline/queue.json").read_text(encoding="utf-8"))
task = next(t for t in queue["gpu_queues"]["server-2"] if t["id"] == "AdaRES-Y18")
assert task["pid"] == 7449 and str(task["gpu"]) == "1"
output = ROOT / ("pipeline/analyses/y18_terminal_" + args.token + ".json")
assert not output.exists() and not output.with_suffix(".log").exists()
program = r'''
import hashlib,json,pathlib,re,subprocess
task=__TASK__
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
def processes():
    p=pathlib.Path('/proc/7449/cmdline')
    original=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
    matching=[]
    for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
        try:cmd=p.read_bytes()
        except OSError:continue
        if b'tools/train.py' in cmd and task['config'].encode() in cmd:
            matching.append(int(p.parent.name))
    return dict(original_cmd=original,matching_y18_pids=matching)
before=processes()
log=pathlib.Path(task['log'])
with log.open('rb') as stream:
    stream.seek(max(0,log.stat().st_size-24000));raw=stream.read(24000)
tail=raw.decode('utf-8',errors='replace')
saved=re.findall(r'save last pth to ([^\s]+/last\.pth)',tail)
rows=[]
if saved:
    base=pathlib.Path(saved[-1]).resolve().parent
    base.relative_to(pathlib.Path('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints').resolve())
    import torch
    for name in ('last.pth','best.pth'):
        p=base/name
        if not p.exists():continue
        oldstat=p.stat()
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        value=torch.load(str(p),map_location='cpu')
        meta=value.get('meta',{})
        rows.append(dict(path=str(p),bytes=oldstat.st_size,sha256=h.hexdigest(),
            epoch=int(meta['epoch']),best_miou=float(meta['best_miou']),
            state_keys={k:k in value for k in ('model','optimizer','scheduler')},
            unchanged_during_read=(oldstat.st_size,oldstat.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)))
        del value
after=processes()
print(json.dumps(dict(id=task['id'],pid=task['pid'],log=str(log),
    processes_before=before,processes_after=after,checkpoints=rows,
    tail_sha256=hashlib.sha256(raw).hexdigest(),tail_last2400=tail[-2400:],
    bounded_traceback='Traceback (most recent call last)' in tail,
    bounded_oom='CUDA out of memory' in tail,
    gpu1_csv=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode(),
    gpu1_compute_pids=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode())))
'''.replace("__TASK__", repr(task))
start = datetime.now().astimezone().isoformat()
result = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
     "songcl@192.168.1.163", "timeout --signal=TERM --kill-after=5s 80 " + server["python"] + " -"],
    input=program.encode("utf-8"), capture_output=True, timeout=90,
)
check_lease()
output.with_suffix(".log").write_bytes(result.stdout + b"\nSTDERR:\n" + result.stderr)
if result.returncode:
    raise RuntimeError("Terminal capture failed; raw log retained")
evidence = json.loads(result.stdout)
evidence.update(token=args.token, observed_start_local=start,
                observed_end_local=datetime.now().astimezone().isoformat(),
                scope="Bounded original log/CPU checkpoint read; GPU1 only. Canonical lifecycle unchanged; no full-log crash or efficacy proof.")
output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print(json.dumps(evidence, indent=2))
print("sha256=" + hashlib.sha256(output.read_bytes()).hexdigest())
