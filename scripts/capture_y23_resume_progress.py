"""Capture fresh native progress for the exact GPU1 Y23 continuation."""

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
    assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])


check_lease()
server = yaml.safe_load((ROOT / "SERVERS.yaml").read_text(encoding="utf-8"))["servers"]["server-2"]
assert server["allowed_gpus"] == [1] and server["host"] == "192.168.1.163"
queue = json.loads((ROOT / "pipeline/queue.json").read_text(encoding="utf-8"))
task = next(t for t in queue["gpu_queues"]["server-2"] if t["id"] == "AdaRES-Y23")
assert task["status"] == "running" and task["launch_verified"] and task["gpu"] == "1"
assert task["resume_attempt_token"] == args.token and task["resume"] == task["resume_checkpoint"]
output = ROOT / ("pipeline/analyses/y23_resume_progress_" + args.token + ".json")
assert not output.exists() and not output.with_suffix(".log").exists()
program = r'''
import hashlib,json,pathlib,re,subprocess,time
task=__TASK__
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
cmd=pathlib.Path('/proc/'+str(task['pid'])+'/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert task['config'] in cmd and '--gpu_ids 1' in cmd
assert '--resume '+task['resume'] in cmd
assert not pathlib.Path('/proc/7449').exists()
log=pathlib.Path(task['log']);captures=[]
for obs in range(2):
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-24000));raw=stream.read(24000)
    text=raw.decode('utf-8',errors='replace')
    matches=re.findall(r'Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)',text)
    assert matches
    epoch,total,step,batches=map(int,matches[-1])
    assert epoch>=5 and total==30
    captures.append(dict(epoch=epoch,total=total,step=step,batches=batches,
        tail_sha256=hashlib.sha256(raw).hexdigest(),bounded_traceback='Traceback' in text,
        bounded_oom='CUDA out of memory' in text))
    if not obs:time.sleep(5)
assert (captures[1]['epoch'],captures[1]['step'])>(captures[0]['epoch'],captures[0]['step'])
assert not any(c['bounded_traceback'] or c['bounded_oom'] for c in captures)
print(json.dumps(dict(id=task['id'],pid=task['pid'],gpu='1',cmdline=cmd,log=str(log),
    resume=task['resume'],captures=captures,launch_verified=True,
    gpu1_csv=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode())))
'''.replace("__TASK__", repr(task))
start = datetime.now().astimezone().isoformat()
response = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "songcl@192.168.1.163", server["python"] + " -"],
    input=program.encode("utf-8"), capture_output=True, timeout=45)
check_lease()
output.with_suffix(".log").write_bytes(response.stdout+b"\nSTDERR:\n"+response.stderr)
if response.returncode:
    raise RuntimeError("Resume capture failed; raw log preserved; do not relaunch")
evidence = json.loads(response.stdout)
evidence.update(token=args.token, observed_start_local=start,
    observed_end_local=datetime.now().astimezone().isoformat(),
    scientific_status="Y23 resumed training; efficacy unknown",
    scope="Two bounded native observations; GPU1 device occupancy, not isolated candidate peak; no bitwise/whole-loader resume equivalence claim. GPU0 not queried.")
output.write_text(json.dumps(evidence,indent=2),encoding="utf-8")
print(json.dumps(evidence,indent=2))
print("sha256="+hashlib.sha256(output.read_bytes()).hexdigest())
