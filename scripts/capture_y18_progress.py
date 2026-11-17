"""Read bounded Y18 native training evidence on the remaining authorized GPU."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
owner = root / "pipeline/headless/.foreground.lockdir/owner.json"
assert json.loads(owner.read_text())["token"] == args.token
queue = json.loads((root / "pipeline/queue.json").read_text(encoding="utf-8"))
task = next(t for t in queue["gpu_queues"]["server-2"] if t["id"] == "AdaRES-Y18")
assert task["status"] == "running" and str(task["gpu"]) == "1"
assert task["launch_verified"] and task["pid"] == 7449
output = root / ("pipeline/analyses/y18_progress_" + args.token + ".json")
assert not output.exists() and not output.with_suffix(".log").exists()
program = r'''
import hashlib,json,pathlib,re,subprocess,time
task=__TASK__
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
cmd=pathlib.Path("/proc/"+str(task["pid"])+"/cmdline").read_bytes().replace(b"\0",b" ").decode()
assert task["config"] in cmd and "--gpu_ids 1" in cmd
path=pathlib.Path(task["log"])
captures=[]
for observation in range(2):
    with path.open("rb") as stream:
        stream.seek(max(0,path.stat().st_size-24000));raw=stream.read(24000)
    text=raw.decode("utf-8",errors="replace")
    matches=re.findall(r"Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)",text)
    assert matches
    epoch,total,step,batches=map(int,matches[-1])
    captures.append(dict(epoch=epoch,total=total,step=step,batches=batches,
        tail1200_sha256=hashlib.sha256(raw[-1200:]).hexdigest(),
        bounded_traceback="Traceback (most recent call last)" in text,
        bounded_oom="CUDA out of memory" in text))
    if not observation:time.sleep(5)
assert (captures[1]["epoch"],captures[1]["step"])>(captures[0]["epoch"],captures[0]["step"])
assert not any(c["bounded_traceback"] or c["bounded_oom"] for c in captures)
print(json.dumps(dict(id=task["id"],pid=task["pid"],gpu="1",config=task["config"],
    log=task["log"],captures=captures,launch_verified=True,
    gpu_csv=subprocess.check_output(["nvidia-smi","-i","1",
        "--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"]).decode())))
'''.replace("__TASK__", repr(task))
start = datetime.now().astimezone().isoformat()
result = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
     "songcl@192.168.1.163", "/home/songcl/.conda/envs/adares_project/bin/python -"],
    input=program.encode(), capture_output=True, timeout=45,
)
assert json.loads(owner.read_text())["token"] == args.token
output.with_suffix(".log").write_bytes(result.stdout + b"\nSTDERR:\n" + result.stderr)
if result.returncode:
    raise RuntimeError("Capture failed; preserved raw log")
evidence = json.loads(result.stdout)
evidence.update(observed_start_local=start, observed_end_local=datetime.now().astimezone().isoformat(),
    token=args.token, scientific_status="Y18 training; Y23 user-paused awaiting Y18 epoch30",
    scope="Two bounded observations; no full-log crash proof or isolated candidate memory. GPU0 not queried.")
output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print(json.dumps(evidence, indent=2))
print("sha256=" + hashlib.sha256(output.read_bytes()).hexdigest())
