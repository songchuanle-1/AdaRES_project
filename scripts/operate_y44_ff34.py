"""Exact tested-file commit, gated GPU1 dispatch and original native progress."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("commit", "admit", "verify"))
parser.add_argument("--token", required=True)
parser.add_argument("--decision")
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 180
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import closure
import pipeline_monitor as monitor

BASE = "pipeline/analyses/"
ID = "AdaRES-Y44"
CONFIG = "configs_song/model_paper5/adares_y44_coordinate_feature_res.yaml"


def evidence(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def git(*arguments):
    return subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false",
         "-C", str(ROOT), *arguments], text=True, encoding="utf-8").strip()


def save(name, value):
    path = BASE + name + "_" + args.token + ".json"
    assert not (ROOT / path).exists()
    (ROOT / path).write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


bundle_path = BASE + "y44_source_bundle_" + args.token + ".json"
bundle = load(bundle_path)
assert bundle["complete"] and bundle["physical_independence"] and bundle["remote_byte_identity"]
paths = [row["path"] for row in bundle["files"]]
assert len(paths) == 7
for row in bundle["files"]:
    assert row["path"].startswith("AdaRES/ipdn/model_paper5/adares_y44_coordinate_feature/") or row["path"] == "AdaRES/" + CONFIG
    assert evidence(row["path"]) == row
review = BASE + "Y44_six_gate_review_ff347697.md"
assert (ROOT / review).is_file()
review_json = load(BASE + "y44_gate_checks_" + args.token + ".json")
assert review_json["complete"] and all(review_json["gates"].values())
terminal = load(BASE + "y42_terminal_ff34_" + args.token + ".json")
protected = dict(terminal["source_hashes"], **terminal["protected_source_hashes"])
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["user"] == "songcl" and server["allowed_gpus"] == [0, 1]
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work_path = ROOT / "pipeline/research_work.json"
    work = load("pipeline/research_work.json")
    candidate = next(c for c in work["candidates"] if c["id"] == ID)
    if args.mode == "commit":
        assert candidate["stage"] == "validation"
        before = git("rev-parse", "HEAD")
        assert before == "6fc7d4e46b6c26a11fede7f12d4e79b3471cc1cc"
        assert not git("diff", "--cached", "--name-only")
        git("add", "--", *paths)
        assert set(git("diff", "--cached", "--name-only").splitlines()) == set(paths)
        output = git("commit", "--only", "-m", "Add Y44 coordinate-feature association confidence", "--", *paths)
        commit = git("rev-parse", "HEAD")
        assert set(git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()) == set(paths)
        assert not git("diff", "--cached", "--name-only")
        for row in bundle["files"]:
            blob = subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + row["path"]])
            assert hashlib.sha256(blob).hexdigest() == row["sha256"]
        path = save("y44_commit_ff34", {"complete": True, "id": ID, "previous_head": before,
                    "code_commit": commit, "changed_paths": paths, "origin_push": False,
                    "output": output, "observed_at_local": datetime.now().astimezone().isoformat()})
        candidate["evidence"].append(evidence(path))
        candidate.update(stage="admission", code_commit=commit,
                         next_action="Bind actual six-gate receipt and decision; canonical GPU1 dispatch and increasing native progress.")
        print(json.dumps({"code_commit": commit, "path": path}))
    else:
        assert args.decision and args.decision.startswith("pipeline/decisions/")
        commit = load(BASE + "y44_commit_ff34_" + args.token + ".json")["code_commit"]
        assert git("rev-parse", "HEAD") == commit
        queue = monitor.load_queue(str(ROOT))
        if args.mode == "admit":
            assert candidate["stage"] == "admission"
            goal_sha = evidence("pipeline/goal.json")["sha256"]
            assert goal_sha == "53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966"
            design = "pipeline/designs/exp_design_Y44_ff347697.md"
            motivation = "pipeline/reviews/motivation_review_Y44_ff347697.md"
            gates = {
                "diagnosis": [BASE + "Y44_diagnosis_prereg_ff347697.md",
                              BASE + "y44_pairing_cpu_ff34_" + args.token + ".json",
                              BASE + "y44_parent_cpu_ff34_" + args.token + ".json",
                              BASE + "y44_native_controls_ff34_v3_" + args.token + ".json"],
                "controls": [BASE + "Y44_native_freeze_ff347697.md",
                             BASE + "y44_native_controls_ff34_v3_" + args.token + ".json",
                             BASE + "y44_initial_loss_ff34_" + args.token + ".json"],
                "native_backward": [BASE + "y44_native_controls_ff34_v3_" + args.token + ".json",
                                    BASE + "y44_physical_native_ff34_" + args.token + ".json"],
                "resources": [BASE + "y44_resources_ff34_" + args.token + ".json", design],
                "independence": [bundle_path, BASE + "y44_parent_cpu_ff34_" + args.token + ".json",
                                 BASE + "y44_physical_native_ff34_" + args.token + ".json"],
                "constraints": [motivation, design, review, BASE + "y44_gate_checks_" + args.token + ".json"],
            }
            receipt = {
                "id": ID, "config": CONFIG, "code_commit": commit, "goal_sha256": goal_sha,
                "decision": evidence(args.decision),
                "gates": {key: {"passed": True, "evidence": [evidence(p) for p in refs]} for key, refs in gates.items()},
                "scientific_status": "admitted_for_falsifiable_training_not_efficacy",
                "frozen_protocol": {"epochs": 30, "seed": 1999, "server": "server-2",
                                    "allowed_gpus": [1], "post_training_design": evidence(design)},
                "limitations": "Initial adverse loss/wrong-control learnability retained; no task benefit, necessity, novelty or human approval. No diagnostic weights deployed.",
            }
            task = {"id": ID, "label": "Y44 all-coordinate feature-association confidence",
                    "config": CONFIG, "status": "waiting", "depends_on": [], "code_commit": commit,
                    "admission_receipt": "pipeline/admissions/" + ID + ".json",
                    "requires_exclusive_ownership": True, "min_epochs": 30, "patience": 30, "total_epochs": 30}
            active = [t for entries in queue["gpu_queues"].values() for t in entries if t["status"] in ("running", "waiting", "claiming")]
            assert len(active) == 1 and active[0]["id"] == "AdaRES-Y43"
            assert active[0]["pid"] == 13155 and str(active[0]["gpu"]) == "0"
            assert not any(t["id"] == ID for entries in queue["gpu_queues"].values() for t in entries)
            payload = {"files": bundle["files"], "protected": protected}
            program = r'''
import hashlib,pathlib,subprocess
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")
cmd=pathlib.Path("/proc/13155/cmdline").read_bytes().split(b"\0")
assert b"configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml" in cmd and cmd[cmd.index(b"--gpu_ids")+1]==b"0"
assert not pathlib.Path("/proc/91215").exists()
for row in P["files"]:
    assert hashlib.sha256((root/row["path"]).read_bytes()).hexdigest()==row["sha256"]
for path,sha in P["protected"].items():
    assert hashlib.sha256((root/"AdaRES"/path).read_bytes()).hexdigest()==sha
print("Y43_AND_IMMUTABLE_SOURCES_VERIFIED")
'''
            response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "songcl@192.168.1.163",
                                       server["python"] + " -c " + shlex.quote("P=" + repr(payload) + "\n" + program)],
                                      capture_output=True, timeout=60)
            assert response.returncode == 0 and response.stdout.decode().strip() == "Y43_AND_IMMUTABLE_SOURCES_VERIFIED"
            original_ssh = monitor.ssh

            def scoped_ssh(host, user, command, *positional, **keywords):
                assert host == server["host"] and user == server["user"]
                if command.startswith("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total"):
                    command = command.replace("nvidia-smi ", "nvidia-smi -i 1 ", 1)
                if "nvidia-smi" in command:
                    assert "nvidia-smi -i 1 " in command
                return original_ssh(host, user, command, *positional, **keywords)

            receipt_path = ROOT / task["admission_receipt"]
            assert not receipt_path.exists()
            receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            closure.admission(ROOT, task)
            preserved_queue = copy.deepcopy(queue)
            queue["gpu_queues"]["server-2"].append(task)
            monitor.save_queue(str(ROOT), queue)
            monitor.ssh = scoped_ssh
            events = []
            try:
                monitor._process_queue(queue, "server-2", server["host"], server["user"], str(ROOT), {"0"}, {"1"},
                                       events, max_tasks_per_gpu=1, remote_code_root=server["remote_code_root"],
                                       current_tasks=1, max_tasks_total=2)
            finally:
                monitor.ssh = original_ssh
            latest = monitor.load_queue(str(ROOT))
            for sid, entries in preserved_queue["gpu_queues"].items():
                assert [t for t in latest["gpu_queues"][sid] if t["id"] != ID] == entries
            selected = next(t for t in latest["gpu_queues"]["server-2"] if t["id"] == ID)
            path = save("y44_dispatch_ff34", {"complete": True, "observed_at_local": datetime.now().astimezone().isoformat(),
                    "task": selected, "events": events, "receipt": evidence(task["admission_receipt"]),
                    "preserved_y43_exact": True, "scope": "Canonical GPU1 dispatch, not repeated preflight."})
            candidate["evidence"].extend([evidence(path), evidence(task["admission_receipt"]), evidence(args.decision)])
            candidate.update(stage="deployment", scientific_status="admitted_not_verified_training",
                             next_action="Verify original Y44 PID/config/GPU1 and increasing native steps; preserve Y43/GPU0; never blindlyrestart.")
            print(json.dumps({"task": selected, "events": events}, indent=2))
        else:
            assert candidate["stage"] == "deployment"
            task = next(t for t in queue["gpu_queues"]["server-2"] if t["id"] == ID)
            assert task["status"] == "running" and str(task["gpu"]) == "1"
            payload = {"task": task, "files": bundle["files"], "protected": protected}
            program = r'''
import hashlib,json,pathlib,re,subprocess,time
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")
code=root/"AdaRES"
t=P["task"]
assert (code/".codex_source_commit").read_text().strip()==t["code_commit"]
for row in P["files"]:
    assert hashlib.sha256((root/row["path"]).read_bytes()).hexdigest()==row["sha256"]
for path,sha in P["protected"].items():
    assert hashlib.sha256((code/path).read_bytes()).hexdigest()==sha
pid=t["pid"]
assert pathlib.Path("/proc/"+str(pid)+"/cwd").resolve()==code
argv=pathlib.Path("/proc/"+str(pid)+"/cmdline").read_bytes().split(b"\0")
assert b"tools/train.py" in argv and t["config"].encode() in argv and argv[argv.index(b"--gpu_ids")+1]==b"1"
old=pathlib.Path("/proc/13155/cmdline").read_bytes()
assert b"configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml" in old
assert old.split(b"\0")[old.split(b"\0").index(b"--gpu_ids")+1]==b"0"
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
        assert time.monotonic()<limit,"No increasing original native progress"
        time.sleep(5)
assert all(c["epochs"]==30 for c in captures)
assert pathlib.Path("/proc/13155/cmdline").read_bytes()==old
used,util=map(int,subprocess.check_output(["nvidia-smi","-i","1","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"]).decode().strip().split(","))
print(json.dumps(dict(complete=True,id="AdaRES-Y44",pid=pid,config=t["config"],log=t["log"],code_commit=t["code_commit"],
    actual_gpu=1,native_steps_increased=True,captures=captures,gpu1_device_memory_mib=used,gpu1_device_utilization=util,
    preserved_y43_pid=13155,preserved_y43_gpu=0,source_hashes_verified=True,
    limitations=["Bounded log, not whole-log crash absence; device occupation is not isolated peak/efficacy."])))
'''
            output = ROOT / (BASE + "y44_launch_ff34_" + args.token + ".json")
            assert not output.exists() and not output.with_suffix(".log").exists()
            started = datetime.now().astimezone().isoformat()
            response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "songcl@192.168.1.163",
                                       server["python"] + " -c " + shlex.quote("P=" + repr(payload) + "\n" + program)],
                                      capture_output=True, timeout=120)
            output.with_suffix(".log").write_bytes(response.stdout + response.stderr)
            assert response.returncode == 0, "Preserve failure; no automatic restart"
            proof = json.loads(response.stdout)
            proof.update(observed_start_local=started, observed_end_local=datetime.now().astimezone().isoformat())
            assert proof["complete"] and proof["native_steps_increased"]
            output.write_text(json.dumps(proof, indent=2), encoding="utf-8")
            current = next(t for t in monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"] if t["id"] == ID)
            assert current["pid"] == proof["pid"] and current["launch_verified"] is True
            candidate["evidence"].append(evidence(output.relative_to(ROOT).as_posix()))
            candidate.update(stage="training", scientific_status="training_not_efficacy",
                             next_action="Preserve original Y44GPU1/Y43GPU0 fixed30. Terminal full9508/141twiceEACH/FIRST/scenes10000seed1999/size/object/harmvsEither/repeat/trainedcost. No retune/blindrestart.")
            print(json.dumps({"path": output.relative_to(ROOT).as_posix(), "proof": proof}, indent=2))
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
finally:
    lock.close()
