"""Foreground-only, guarded continuation of the user-paused Y23 on GPU1."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SKILL = "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline"
sys.path.insert(0, SKILL)
import closure
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
parser.add_argument("--check-only", action="store_true")
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None, "Pipeline writer busy"
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["user"] == "songcl"
assert list(server["allowed_gpus"]) == [1], "GPU0 must remain reserved"
queue = monitor.load_queue(str(ROOT))
tasks = queue["gpu_queues"]["server-2"]
y18 = next((t for t in tasks if t["id"] == "AdaRES-Y18"), None)
if y18 is None:
    y18 = monitor.load_experiment(str(ROOT), "AdaRES-Y18")
assert y18 is not None, "Missing canonical Y18 lifecycle"
y23 = next(t for t in tasks if t["id"] == "AdaRES-Y23")
assert y23["status"] == "paused", "Do not replay a claiming/running resume"
out = ROOT / ("pipeline/analyses/y23_resume_guard_" + args.token + ".json")
assert not out.exists(), "Guard evidence exists; inspect it instead of replaying"
if y18["status"] != "done_complete":
    result = dict(result="blocked_dependency", y18_status=y18["status"],
                  next_action="Wait for Y18 fixed30epoch done_complete; GPU0 remains reserved",
                  observed_local=datetime.now().astimezone().isoformat())
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    print("sha256=" + hashlib.sha256(out.read_bytes()).hexdigest())
    lock.close()
    sys.exit(0)
closure.admission(str(ROOT), y23)
assert closure.git(str(ROOT), "rev-parse", "HEAD") == y23["code_commit"]
checkpoint = y23["resume_checkpoint"]
expected_hash = "57aafa1f85f7214cf85c91ce525e373716d3a9dcf1b78164e626a4246dd9e685"
bundle = json.loads((ROOT / "pipeline/analyses/y23_source_bundle_d20c71e4.json").read_text())
files = list(bundle["files"])
train_path = ROOT / "AdaRES/tools/train.py"
files.append(dict(path="AdaRES/tools/train.py", sha256=hashlib.sha256(train_path.read_bytes()).hexdigest()))
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log = server["remote_code_root"] + "/logs/resume_adares_y23_valid_token_support_res_" + stamp + ".log"
payload = dict(server=server, checkpoint=checkpoint, checkpoint_sha256=expected_hash,
               config=y23["config"], commit=y23["code_commit"], files=files,
               log=log, check_only=args.check_only)
program = r'''
import hashlib,json,os,pathlib,subprocess
payload=__PAYLOAD__
s=payload["server"]
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
root=pathlib.Path(s["remote_code_root"]).resolve()
assert (root/".codex_source_commit").read_text().strip()==payload["commit"]
for entry in payload["files"]:
    p=root.parent/entry["path"]
    assert hashlib.sha256(p.read_bytes()).hexdigest()==entry["sha256"],str(p)
assert not pathlib.Path("/proc/7449").exists(),"Original Y18 PID remains"
for p in pathlib.Path("/proc").glob("[0-9]*/cmdline"):
    try:cmd=p.read_bytes()
    except OSError:continue
    if b"tools/train.py" in cmd:
        assert b"adares_y18_point_measure_res.yaml" not in cmd,"Y18 still running"
        assert payload["config"].encode() not in cmd,"Y23 already running"
csv=subprocess.check_output(["nvidia-smi","-i","1","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"]).decode()
memory,util=map(int,csv.strip().split(","))
assert memory<1024 and util<10,"GPU1 occupied"
gpu_uuid=subprocess.check_output(["nvidia-smi","-i","1","--query-gpu=uuid","--format=csv,noheader"]).decode().strip()
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"]).decode()
assert gpu_uuid not in apps,"GPU1 has a compute process"
p=pathlib.Path(payload["checkpoint"])
h=hashlib.sha256()
with p.open("rb") as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
assert h.hexdigest()==payload["checkpoint_sha256"],"Checkpoint changed"
import torch
y18_log=pathlib.Path("/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y18_point_measure_res_20261001_233715.log")
with y18_log.open("rb") as stream:
    stream.seek(max(0,y18_log.stat().st_size-24000))
    terminal_tail=stream.read().decode("utf-8",errors="replace")
import re
saved=re.findall(r"save last pth to ([^\s]+/last\.pth)",terminal_tail)
assert saved,"Y18 original log has no terminal last checkpoint"
y18_checkpoint=pathlib.Path(saved[-1])
y18_checkpoint.resolve().relative_to((root.parent/"AdaRES_checkpoints").resolve())
y18_meta=torch.load(str(y18_checkpoint),map_location="cpu")["meta"]
assert int(y18_meta["epoch"])==30,"Y18 has not saved epoch30"
c=torch.load(str(p),map_location="cpu")
assert int(c["meta"]["epoch"])==4 and all(k in c for k in ("model","optimizer","scheduler"))
result=dict(result="ready",checkpoint=str(p),checkpoint_sha256=h.hexdigest(),
    resume_epoch=5,total_epochs=30,gpu="1",gpu_memory_before=memory)
if not payload["check_only"]:
    # No broad synchronization or YAML rewrite while Y18's artifacts are retained.
    env=os.environ.copy()
    env["PYTHONPATH"]=":".join([str(root),str(root/"ipdn/lib"),s["site_packages"],s["pointnet_egg"]])
    env["PYTHONDONTWRITEBYTECODE"]="1"
    with open(payload["log"],"xb",buffering=0) as stream:
        proc=subprocess.Popen([s["python"],"-S","tools/train.py",payload["config"],
            "--num_gpus","1","--gpu_ids","1","--resume",str(p)],
            cwd=str(root),env=env,stdin=subprocess.DEVNULL,stdout=stream,
            stderr=subprocess.STDOUT,start_new_session=True)
    result.update(result="launched_pending_progress",pid=proc.pid,log=payload["log"])
print(json.dumps(result))
'''.replace("__PAYLOAD__", repr(payload))
if not args.check_only:
    y23.update(status="claiming", resume_attempt_token=args.token,
               launch_verified=False, launch_verification="pending_progress")
    monitor.save_queue(str(ROOT), queue)
r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                    "songcl@192.168.1.163", server["python"] + " -"],
                   input=program.encode(), capture_output=True, timeout=90)
out.with_suffix(".log").write_bytes(r.stdout + b"\nSTDERR:\n" + r.stderr)
if r.returncode:
    raise RuntimeError("Resume guard/launch failed; keep claiming if launch uncertain, inspect raw log, do not relaunch")
result = json.loads(r.stdout)
if not args.check_only:
    y23.update(pid=result["pid"], log=result["log"], gpu="1", server="server-2",
               resume=checkpoint, launch_verified=False)
    monitor.save_queue(str(ROOT), queue)
    verified = closure.verify_progress(server, y23, monitor.ssh)
    y23.update(status="running", launch_verified=verified,
               launch_verification="training_progress" if verified else "pending_progress")
    monitor.save_queue(str(ROOT), queue)
    running = monitor.load_running(str(ROOT))
    running.setdefault("server-2", {})["1"] = "AdaRES-Y23"
    monitor.save_running(str(ROOT), running)
    experiment = monitor.load_experiment(str(ROOT), "AdaRES-Y23")
    experiment.update(status="running", pid=result["pid"], log=result["log"], gpu="1",
                      resume=checkpoint, launch_verified=verified,
                      launch_verification=y23["launch_verification"])
    experiment.setdefault("history", []).append(dict(ts=datetime.now().astimezone().isoformat(),
        action="resumed_after_y18", gpu="1", checkpoint=checkpoint, pid=result["pid"]))
    monitor.save_experiment(str(ROOT), experiment)
    result["launch_verified"] = verified
result["observed_local"] = datetime.now().astimezone().isoformat()
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
lock.close()
print(json.dumps(result))
print("sha256=" + hashlib.sha256(out.read_bytes()).hexdigest())
if not args.check_only and not result["launch_verified"]:
    raise RuntimeError("Resume launched but progress unverified; reconcile exact PID/log, never restart blindly")
