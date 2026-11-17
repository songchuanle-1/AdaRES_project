"""Apply the user's Y23 pause without touching the Y18 process."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "a865bcc791474a39be4cc04af13ccb52"
assert json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())["token"] == TOKEN
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

lock = monitor.acquire_lock(str(ROOT))
assert lock is not None, "Another pipeline writer holds the project lock"
program = r'''
import hashlib,json,os,pathlib,re,signal,subprocess,time
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
pid=125903
proc=pathlib.Path("/proc")/str(pid)
cmd=proc.joinpath("cmdline").read_bytes().replace(b"\0",b" ").decode()
assert "tools/train.py configs_song/model_paper5/adares_y23_valid_token_support_res.yaml" in cmd
assert "--gpu_ids 0" in cmd
other=pathlib.Path("/proc/7449/cmdline").read_bytes()
assert b"adares_y18_point_measure_res.yaml" in other
checkpoint=pathlib.Path("/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_y23_valid_token_support_res/20261001_225332/last.pth")
assert checkpoint.is_file() and checkpoint.stat().st_size>100000000
log=pathlib.Path("/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y23_valid_token_support_res_20261002_112523.log")
with log.open("rb") as stream:
    stream.seek(max(0,log.stat().st_size-24000));tail=stream.read()
steps=re.findall(r"Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)",tail.decode("utf8",errors="replace"))
children=[]
for p in pathlib.Path("/proc").glob("[0-9]*"):
    try:
        status=p.joinpath("status").read_text()
        if int(re.search(r"^PPid:\s*(\d+)",status,re.M).group(1))==pid:
            children.append((int(p.name),p.joinpath("stat").read_text().split()[-30:]))
    except (OSError,ValueError,AttributeError):pass
os.kill(pid,signal.SIGTERM)
for _ in range(100):
    if not proc.exists():break
    time.sleep(.1)
assert not proc.exists(),"Y23 parent did not exit after SIGTERM; no broad kill"
for child,_ in children:
    p=pathlib.Path("/proc")/str(child)
    if p.exists():
        try:
            ppid=int(re.search(r"^PPid:\s*(\d+)",p.joinpath("status").read_text(),re.M).group(1))
            args=p.joinpath("cmdline").read_bytes()
            if ppid==1 and b"adares_y23_valid_token_support_res.yaml" in args:
                os.kill(child,signal.SIGTERM)
        except OSError:pass
time.sleep(2)
assert pathlib.Path("/proc/7449/cmdline").read_bytes()==other
gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,utilization.gpu","--format=csv,noheader,nounits"]).decode()
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_gpu_memory","--format=csv,noheader,nounits"]).decode()
assert "125903" not in apps
print(json.dumps(dict(stopped_pid=pid,signal="SIGTERM",y18_pid=7449,y18_command_unchanged=True,
    last_train=steps[-1] if steps else None,checkpoint=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
    checkpoint_mtime=checkpoint.stat().st_mtime,original_log=str(log),
    tail_sha256=hashlib.sha256(tail).hexdigest(),gpu_csv=gpu,compute_apps=apps)))
'''
start = datetime.now().astimezone().isoformat()
result = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
     "songcl@192.168.1.163", "/home/songcl/.conda/envs/adares_project/bin/python -"],
    input=program.encode(), capture_output=True, timeout=45,
)
out = ROOT / "pipeline/analyses/y23_pause_a865bcc7.json"
assert not out.exists()
out.with_suffix(".log").write_bytes(result.stdout + b"\nSTDERR:\n" + result.stderr)
if result.returncode:
    raise RuntimeError("Pause verification failed; preserved raw log")
evidence = json.loads(result.stdout)
evidence.update(observed_start_local=start, observed_end_local=datetime.now().astimezone().isoformat())
out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
queue = monitor.load_queue(str(ROOT))
task = next(t for t in queue["gpu_queues"]["server-2"] if t["id"] == "AdaRES-Y23")
task.update(status="paused", dispatch_hold="User pause: resume only on GPU1 after AdaRES-Y18 completes",
            resume_checkpoint=evidence["checkpoint"], resume_gpu="1", depends_on=["AdaRES-Y18"],
            paused_at=evidence["observed_end_local"], pause_evidence="pipeline/analyses/y23_pause_a865bcc7.json")
monitor.save_queue(str(ROOT), queue)
running = monitor.load_running(str(ROOT))
assert running["server-2"].get("0") == "AdaRES-Y23"
running["server-2"].pop("0")
monitor.save_running(str(ROOT), running)
experiment = monitor.load_experiment(str(ROOT), "AdaRES-Y23")
experiment.update(status="paused", paused_at=evidence["observed_end_local"],
                  resume_checkpoint=evidence["checkpoint"], resume_gpu="1",
                  dispatch_hold=task["dispatch_hold"])
experiment.setdefault("history", []).append(dict(ts=evidence["observed_end_local"],
    action="user_paused", gpu="0", pid=125903, resume_after="AdaRES-Y18", resume_gpu="1"))
monitor.save_experiment(str(ROOT), experiment)
lock.close()
print(json.dumps(evidence, indent=2))
print("sha256=" + hashlib.sha256(out.read_bytes()).hexdigest())
