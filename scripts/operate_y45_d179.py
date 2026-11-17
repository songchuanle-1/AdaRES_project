"""Exact seven-file commit and canonical GPU1-only deployment of admitted Y45."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "392a5a9f5fe242d7a3109a50e6060fda"
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import closure
import pipeline_monitor as monitor
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("commit", "admit", "verify"))
parser.add_argument("--token", required=True)
parser.add_argument("--decision")
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 240
ID = "AdaRES-Y45"
CONFIG = "configs_song/model_paper5/adares_y45_metric_decomposition_res.yaml"


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def evidence(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


def git(*arguments):
    return subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "-C", str(ROOT), *arguments],
        text=True, encoding="utf-8").strip()


def save(stem, data):
    path = "pipeline/analyses/" + stem + "_" + args.token + ".json"
    assert not (ROOT / path).exists()
    (ROOT / path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


bundle_path = "pipeline/analyses/y45_source_bundle_392a_" + PRIOR + ".json"
bundle = load(bundle_path)
assert bundle["complete"] and bundle["physical_independence"] and bundle["remote_byte_identity"]
checks_path = "pipeline/analyses/y45_gate_checks_d179_" + args.token + ".json"
checks = load(checks_path)
assert checks["complete"] and all(checks["gates"].values())
paths = [row["path"] for row in bundle["files"]]
assert len(paths) == 7
for row in bundle["files"]:
    assert row["path"].startswith("AdaRES/ipdn/model_paper5/adares_y45_metric_decomposition/") or row["path"] == "AdaRES/" + CONFIG
    assert evidence(row["path"]) == row
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["user"] == "songcl" and server["allowed_gpus"] == [0, 1]
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    work = load("pipeline/research_work.json")
    candidate = next(c for c in work["candidates"] if c["id"] == ID)
    if args.mode == "commit":
        assert candidate["stage"] == "validation"
        before = git("rev-parse", "HEAD")
        assert before == "c65ad2fcba4ef10b71ea6ae58ed01109568bd0cd"
        assert not git("diff", "--cached", "--name-only")
        git("add", "--", *paths)
        assert set(git("diff", "--cached", "--name-only").splitlines()) == set(paths)
        output = git("commit", "--only", "-m", "Add Y45 native metric error decomposition", "--", *paths)
        commit = git("rev-parse", "HEAD")
        assert set(git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()) == set(paths)
        assert not git("diff", "--cached", "--name-only")
        for row in bundle["files"]:
            blob = subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + row["path"]])
            assert hashlib.sha256(blob).hexdigest() == row["sha256"]
        path = save("y45_commit_d179", dict(
            complete=True, id=ID, previous_head=before, code_commit=commit,
            changed_paths=paths, output=output, origin_push=False,
            observed_at_local=datetime.now().astimezone().isoformat()))
        candidate["evidence"].append(evidence(path))
        candidate.update(stage="admission", code_commit=commit,
                         next_action="Bind reviewed gates/decision to the exact commit, dispatch canonically on idle authorized GPU1, verify increasing native steps.")
        print(json.dumps(dict(code_commit=commit, path=path)))
    else:
        assert args.decision and args.decision.startswith("pipeline/decisions/")
        commit = load("pipeline/analyses/y45_commit_d179_" + args.token + ".json")["code_commit"]
        assert git("rev-parse", "HEAD") == commit
        if args.mode == "admit":
            assert candidate["stage"] == "admission"
            queue = monitor.load_queue(str(ROOT))
            active = [t for entries in queue["gpu_queues"].values() for t in entries
                      if t["status"] in ("running", "waiting", "claiming")]
            assert not active and not any(t["id"] == ID for entries in queue["gpu_queues"].values() for t in entries)
            proof = r'''
import hashlib,pathlib,subprocess
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")
for row in P["files"]:
    assert hashlib.sha256((root/row["path"]).read_bytes()).hexdigest()==row["sha256"]
used,util=map(int,subprocess.check_output(["nvidia-smi","-i","1","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"]).decode().strip().split(","))
assert used<1024 and util<10
assert not subprocess.check_output(["nvidia-smi","-i","1","--query-compute-apps=pid","--format=csv,noheader,nounits"]).decode().strip()
print("SOURCE_AND_GPU1_IDLE_VERIFIED")
'''
            response = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "songcl@192.168.1.163",
                 server["python"] + " -c " + shlex.quote("P=" + repr(dict(files=bundle["files"])) + "\n" + proof)],
                capture_output=True, timeout=45)
            assert response.returncode == 0 and response.stdout.decode().strip() == "SOURCE_AND_GPU1_IDLE_VERIFIED"
            receipt = dict(
                id=ID, config=CONFIG, code_commit=commit,
                goal_sha256=evidence("pipeline/goal.json")["sha256"],
                decision=evidence(args.decision),
                gates={key: dict(passed=True, evidence=refs + [checks["review"]])
                       for key, refs in checks["evidence"].items()},
                scientific_status="admitted_for_falsifiable_training_not_efficacy",
                frozen_protocol=dict(epochs=30, seed=1999, server="server-2", allowed_gpus=[1],
                                     post_training_design=evidence("pipeline/designs/exp_design_Y45_392a.md")),
                limitations="Native local objective conflicts, wrong-control learnability, initial losses and v1 failures retained. No efficacy/necessity/novelty/Pareto/global bitwise/Phase2 claim; no diagnostic weights deployed.")
            receipt_path = "pipeline/admissions/" + ID + ".json"
            assert not (ROOT / receipt_path).exists()
            (ROOT / receipt_path).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            task = dict(id=ID, label="Y45 point-confusion decomposition confidence", config=CONFIG,
                        status="waiting", depends_on=[], gpu=1, code_commit=commit,
                        admission_receipt=receipt_path, requires_exclusive_ownership=True,
                        min_epochs=30, patience=30, total_epochs=30)
            closure.admission(ROOT, task)
            queue["gpu_queues"]["server-2"].append(task)
            monitor.save_queue(str(ROOT), queue)
            original_ssh = monitor.ssh

            def scoped_ssh(host, user, command, *positional, **keywords):
                assert host == server["host"] and user == server["user"]
                if command.startswith("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total"):
                    command = command.replace("nvidia-smi ", "nvidia-smi -i 1 ", 1)
                if "nvidia-smi" in command:
                    assert "nvidia-smi -i 1 " in command
                return original_ssh(host, user, command, *positional, **keywords)

            monitor.ssh = scoped_ssh
            events = []
            try:
                monitor._process_queue(queue, "server-2", server["host"], server["user"],
                                       str(ROOT), {"0"}, {"1"}, events, max_tasks_per_gpu=1,
                                       remote_code_root=server["remote_code_root"],
                                       current_tasks=0, max_tasks_total=2)
            finally:
                monitor.ssh = original_ssh
            selected = next(t for t in monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"] if t["id"] == ID)
            path = save("y45_dispatch_d179", dict(
                complete=True, id=ID, task=selected, events=events, receipt=evidence(receipt_path),
                observed_at_local=datetime.now().astimezone().isoformat(),
                scope="Canonical GPU1-only dispatcher; no repeated preflight or GPU0 operation."))
            candidate["evidence"].extend([evidence(path), evidence(receipt_path), evidence(args.decision)])
            candidate.update(stage="deployment", admission_receipt_issued=True,
                             scientific_status="admitted_launch_not_yet_verified",
                             next_action="Verify original Y45 PID/config/GPU1/log and increasing native steps; do not relaunch if uncertain.")
            print(json.dumps(dict(task=selected, events=events), indent=2))
        else:
            assert candidate["stage"] == "deployment"
            task = next(t for t in monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"] if t["id"] == ID)
            assert task["status"] == "running" and str(task["gpu"]) == "1"
            program = r'''
import hashlib,json,pathlib,re,subprocess,time
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")
code=root/"AdaRES"
t=P["task"]
assert (code/".codex_source_commit").read_text().strip()==t["code_commit"]
for row in P["files"]:
    assert hashlib.sha256((root/row["path"]).read_bytes()).hexdigest()==row["sha256"]
pid=t["pid"]
assert pathlib.Path("/proc/"+str(pid)+"/cwd").resolve()==code
argv=pathlib.Path("/proc/"+str(pid)+"/cmdline").read_bytes().split(b"\0")
assert b"tools/train.py" in argv and t["config"].encode() in argv
assert argv[argv.index(b"--gpu_ids")+1]==b"1"
captures=[]
limit=time.monotonic()+90
while len(captures)<2:
    assert pathlib.Path("/proc/"+str(pid)).exists()
    with open(t["log"],"rb") as stream:
        stream.seek(0,2)
        stream.seek(max(0,stream.tell()-24000))
        text=stream.read().decode("utf-8",errors="replace")
    assert not any(s in text for s in ("Traceback","CUDA out of memory","RuntimeError"))
    steps=[tuple(map(int,m)) for m in re.findall(r"Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)",text)]
    if steps:
        epoch,total,step,totalsteps=steps[-1]
        if not captures or (epoch,step)>(captures[0]["epoch"],captures[0]["step"]):
            captures.append(dict(epoch=epoch,epochs=total,step=step,total_steps=totalsteps))
    if len(captures)<2:
        assert time.monotonic()<limit,"No increasing native progress"
        time.sleep(5)
assert all(c["epochs"]==30 for c in captures)
print(json.dumps(dict(complete=True,id="AdaRES-Y45",pid=pid,gpu=1,config=t["config"],log=t["log"],code_commit=t["code_commit"],native_steps_increased=True,captures=captures,source_hashes_verified=True,limitations=["Bounded native steps, not efficacy or whole-log crash absence."])))
'''
            started = datetime.now().astimezone().isoformat()
            response = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "songcl@192.168.1.163",
                 server["python"] + " -c " + shlex.quote("P=" + repr(dict(task=task, files=bundle["files"])) + "\n" + program)],
                capture_output=True, timeout=120)
            raw_path = ROOT / ("pipeline/analyses/y45_launch_d179_" + args.token + ".log")
            assert not raw_path.exists()
            raw_path.write_bytes(response.stdout + response.stderr)
            assert response.returncode == 0, "Preserve failure and reconcile, never blindly restart"
            proof = json.loads(response.stdout)
            proof.update(observed_start_local=started, observed_end_local=datetime.now().astimezone().isoformat())
            assert proof["complete"] and proof["native_steps_increased"]
            current = next(t for t in monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"] if t["id"] == ID)
            assert current["pid"] == proof["pid"] and current["launch_verified"] is True
            path = save("y45_launch_d179", proof)
            candidate["evidence"].append(evidence(path))
            candidate.update(stage="training", scientific_status="training_not_efficacy",
                             next_action="Preserve SAME Y45 GPU1/releasedSPFormer/seed1999/fixed30; reconcile only real changes. Terminal full9508/141 twice EACH/FIRST/W2+B1/scene+subgroup harm/repeat/trained cost; no retune or blind restart.")
            print(json.dumps(proof, indent=2))
    (ROOT / "pipeline/research_work.json").write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
finally:
    writer.close()
