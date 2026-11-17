"""Execute one lease-bound project probe, preserving logs before completion."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("program")
parser.add_argument("--token", required=True)
parser.add_argument("--seconds", type=int, default=300)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--gpu-id", type=int, choices=(0, 1))
args = parser.parse_args()
assert re.fullmatch("[a-z0-9_]{1,64}", args.name)
assert 10 <= args.seconds <= 900
owner_path = ROOT / "pipeline/headless/.foreground.lockdir/owner.json"


def check_lease(reserve):
    owner = json.loads(owner_path.read_text())
    assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > reserve


check_lease(args.seconds + 70)
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["user"] == "songcl"
allowed_gpus = server["allowed_gpus"]
assert allowed_gpus and set(allowed_gpus).issubset({0, 1})
assert args.gpu or args.gpu_id is None, "--gpu-id requires --gpu"
gpu_id = (1 if args.gpu_id is None else args.gpu_id) if args.gpu else None
assert gpu_id is None or gpu_id in allowed_gpus, "Diagnostic GPU not authorized"
source = Path(args.program).resolve()
assert ROOT in source.parents and source.suffix == ".py"
program = source.read_text(encoding="utf-8")
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
output = ROOT / ("pipeline/analyses/" + args.name + "_" + args.token + ".json")
remote_output = "/tmp/adares_" + args.name + "_" + args.token + ".json"
assert not output.exists() and not output.with_suffix(".log").exists()
payload = dict(output=remote_output, token=args.token, root=server["remote_code_root"],
               source_sha256=source_hash, physical_gpu=gpu_id)
prefix = (
    "import os,sys,pathlib,json,subprocess\n"
    "P=" + repr(payload) + "\n"
    "assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'\n"
    "assert not pathlib.Path(P['output']).exists()\n"
    "os.environ['PYTHONDONTWRITEBYTECODE']='1'\n"
    "os.environ['CUDA_VISIBLE_DEVICES']=" + repr(str(gpu_id) if args.gpu else "") + "\n"
    "os.chdir(P['root'])\n"
    "sys.path[:0]=" + repr([server["remote_code_root"], server["remote_code_root"] + "/tools",
                            server["remote_code_root"] + "/ipdn/lib",
                            server["site_packages"], server["pointnet_egg"]]) + "\n"
)
if args.gpu:
    gpu_lock = ("/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock"
                if gpu_id == 1 else
                "/bak_home/songcl/Project/AdaRES_project/.foreground_gpu0_diagnostic.lock")
    prefix += (
        "import fcntl\n"
        "gpu_lease=open(" + repr(gpu_lock) + ",'a')\n"
        "fcntl.flock(gpu_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        "used,util=[int(v.strip()) for v in subprocess.check_output(['nvidia-smi','-i'," + repr(str(gpu_id)) + ","
        "'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')]\n"
        "assert used<1024 and util<10,'Diagnostic GPU occupied'\n"
        "compute_processes=subprocess.check_output(['nvidia-smi','-i'," + repr(str(gpu_id)) + ","
        "'--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()\n"
        "assert not compute_processes,'Diagnostic GPU has compute processes'\n"
    )
compile(prefix + program, str(source), "exec")
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None, "Pipeline writer busy"
start = datetime.now().astimezone().isoformat()
try:
    with output.with_suffix(".log").open("xb") as log:
        response = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "songcl@192.168.1.163",
             "timeout --signal=TERM --kill-after=10s " + str(args.seconds) + " " + server["python"] + " -S -"],
            input=(prefix + program).encode("utf-8"), stdout=log, stderr=subprocess.STDOUT,
            timeout=args.seconds + 25)
    check_lease(30)
    assert response.returncode == 0, "Probe failed; preserve raw log"
    subprocess.run(["scp", "-q", "songcl@192.168.1.163:" + remote_output, str(output)], check=True, timeout=30)
    result = json.loads(output.read_text())
    assert result["complete"], "Incomplete output is not a completed probe"
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                         program_sha256=source_hash, observed_start_local=start,
                         observed_end_local=datetime.now().astimezone().isoformat(),
                         summary={k: v for k, v in result.items() if k not in ("rows", "cases")}), indent=2))
finally:
    lock.close()
