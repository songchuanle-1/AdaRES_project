"""Capture bounded native progress without modifying running experiments."""

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
assert json.loads(owner.read_text(encoding="utf-8"))["token"] == args.token
queue = json.loads((root / "pipeline/queue.json").read_text(encoding="utf-8"))
tasks = [
    task for task in queue["gpu_queues"]["server-2"]
    if task["id"] in ("AdaRES-Y18", "AdaRES-Y23")
]
assert len(tasks) == 2
assert all(task["status"] == "running" and task["launch_verified"] for task in tasks)
assert {str(task["gpu"]) for task in tasks} == {"0", "1"}
output = root / ("pipeline/analyses/native_progress_" + args.token + ".json")
log = output.with_suffix(".log")
assert not output.exists() and not log.exists()
program = r'''
import datetime, hashlib, json, pathlib, re, subprocess, time
tasks = __TASKS__
hostname = subprocess.check_output(["hostname"]).decode().strip()
assert hostname == "dl01-MS-7C60"
rows = []
for task in tasks:
    pid = str(task["pid"])
    cmd = pathlib.Path("/proc/" + pid + "/cmdline").read_bytes()
    cmd = cmd.replace(b"\0", b" ").decode()
    assert task["config"] in cmd and "--gpu_ids " + str(task["gpu"]) in cmd
    captures = []
    path = pathlib.Path(task["log"])
    for observation in range(2):
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 24000))
            raw = stream.read(24000)
        text = raw.decode("utf-8", errors="replace")
        matches = re.findall(r"Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)", text)
        assert matches, "No native steps: " + task["id"]
        epoch, total, step, batches = map(int, matches[-1])
        captures.append(dict(epoch=epoch, total=total, step=step, batches=batches,
            remote_time=datetime.datetime.now().astimezone().isoformat(),
            tail1200_sha256=hashlib.sha256(raw[-1200:]).hexdigest(),
            bounded_traceback="Traceback (most recent call last)" in text,
            bounded_oom="CUDA out of memory" in text))
        if not observation:
            time.sleep(5)
    assert (captures[1]["epoch"], captures[1]["step"]) > (captures[0]["epoch"], captures[0]["step"]), task["id"]
    assert not any(c["bounded_traceback"] or c["bounded_oom"] for c in captures)
    rows.append(dict(id=task["id"], pid=task["pid"], gpu=task["gpu"],
        config=task["config"], log=task["log"], captures=captures,
        launch_verified=True))
print(json.dumps(dict(hostname=hostname, tasks=rows,
    gpu_csv=subprocess.check_output(["nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"]).decode())))
'''.replace("__TASKS__", repr(tasks))
start = datetime.now().astimezone().isoformat()
result = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
     "songcl@192.168.1.163", "/home/songcl/.conda/envs/adares_project/bin/python -"],
    input=program.encode("utf-8"), capture_output=True, timeout=60,
)
assert json.loads(owner.read_text(encoding="utf-8"))["token"] == args.token
log.write_bytes(result.stdout + b"\nSTDERR:\n" + result.stderr)
if result.returncode:
    raise RuntimeError("Capture failed; retained " + str(log))
evidence = json.loads(result.stdout)
evidence.update(
    token=args.token,
    observed_start_local=start,
    observed_end_local=datetime.now().astimezone().isoformat(),
    scientific_status="Y18/Y23 training; final efficacy unknown",
    scope="Two bounded native observations per original PID/log; not full-log crash proof. Device memory is not isolated candidate peak.",
)
output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print(json.dumps(evidence, indent=2))
print("sha256=" + hashlib.sha256(output.read_bytes()).hexdigest())
