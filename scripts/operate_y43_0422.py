"""Exact tested-file commit, canonical GPU0 admission, and independent native progress."""

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
ID = "AdaRES-Y43"
CONFIG = "configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml"


def evidence(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


def load(path):
    return json.loads((ROOT / path).read_text())


def git(*arguments):
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


def save(name, value):
    path = BASE + name + "_" + args.token + ".json"
    assert not (ROOT / path).exists()
    (ROOT / path).write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


bundle_path = BASE + "y43_source_bundle_" + args.token + ".json"
bundle = load(bundle_path)
assert bundle["complete"] and bundle["physical_independence"] and bundle["remote_byte_identity"]
paths = [item["path"] for item in bundle["files"]]
assert len(paths) == 7
for item in bundle["files"]:
    assert item["path"].startswith("AdaRES/ipdn/model_paper5/adares_y43_point_asymmetry/") or item["path"] == "AdaRES/" + CONFIG
    assert evidence(item["path"]) == item
review = BASE + "Y43_six_gate_review_0422d0d8.md"
assert (ROOT / review).is_file()
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["user"] == "songcl" and server["allowed_gpus"] == [0, 1]
terminal = load(BASE + "y41_terminal_0422_" + args.token + ".json")
assert terminal["complete"]
protected = ROOT / "pipeline/admissions/AdaRES-Y42.json"
old_receipt = load(protected.relative_to(ROOT).as_posix())
old_bundle = load(old_receipt["gates"]["independence"]["evidence"][0]["path"])
assert len(old_bundle["files"]) == 7

lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work_path = ROOT / "pipeline/research_work.json"
    work = load("pipeline/research_work.json")
    candidate = next(c for c in work["candidates"] if c["id"] == ID)
    if args.mode == "commit":
        assert candidate["stage"] == "validation"
        before = git("rev-parse", "HEAD")
        assert before == "243e0b6827073831e0d68090857755d575efbf16"
        assert not git("diff", "--cached", "--name-only"), "Do not alter unrelated staged changes"
        git("add", "--", *paths)
        assert set(git("diff", "--cached", "--name-only").splitlines()) == set(paths)
        output = git("commit", "--only", "-m", "Add Y43 point-distribution asymmetry confidence", "--", *paths)
        commit = git("rev-parse", "HEAD")
        assert set(git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()) == set(paths)
        assert not git("diff", "--cached", "--name-only")
        for item in bundle["files"]:
            blob = subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + item["path"]])
            assert hashlib.sha256(blob).hexdigest() == item["sha256"]
        path = save("y43_commit_0422", {
            "complete": True, "id": ID, "previous_head": before, "code_commit": commit,
            "changed_paths": paths, "origin_push": False, "output": output,
            "observed_at_local": datetime.now().astimezone().isoformat()})
        candidate["evidence"].append(evidence(path))
        candidate.update(stage="admission", code_commit=commit,
                         next_action="Bind actual gate receipt/decision; canonical GPU0-only dispatch and independent increasing steps.")
        print(json.dumps({"code_commit": commit, "path": path}))
    else:
        assert args.decision and args.decision.startswith("pipeline/decisions/")
        commit = load(BASE + "y43_commit_0422_" + args.token + ".json")["code_commit"]
        assert git("rev-parse", "HEAD") == commit
        queue = monitor.load_queue(str(ROOT))
        if args.mode == "admit":
            assert candidate["stage"] == "admission"
            goal_sha = evidence("pipeline/goal.json")["sha256"]
            assert goal_sha == "53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966"
            design = "pipeline/designs/exp_design_Y43_0422d0d8.md"
            motivation = "pipeline/reviews/motivation_review_Y43_0422d0d8.md"
            gates = {
                "diagnosis": [BASE + "Y43_diagnosis_prereg_0422d0d8.md",
                              BASE + "y43_reflection_cpu_0422_" + args.token + ".json",
                              BASE + "y43_extract_readout_cpu_0422_" + args.token + ".json",
                              BASE + "y43_native_controls_0422_v2_" + args.token + ".json"],
                "controls": [BASE + "y43_extract_readout_cpu_0422_" + args.token + ".json",
                             BASE + "y43_native_controls_0422_v2_" + args.token + ".json",
                             BASE + "y43_initial_loss_0422_" + args.token + ".json"],
                "native_backward": [BASE + "y43_native_controls_0422_v2_" + args.token + ".json",
                                    BASE + "y43_physical_native_0422_" + args.token + ".json"],
                "resources": [BASE + "y43_resources_0422_" + args.token + ".json", design],
                "independence": [bundle_path, BASE + "y43_parent_semantics_0422_" + args.token + ".json",
                                 BASE + "y43_physical_native_0422_" + args.token + ".json"],
                "constraints": [motivation, design, review],
            }
            receipt = {
                "id": ID, "config": CONFIG, "code_commit": commit,
                "goal_sha256": goal_sha, "decision": evidence(args.decision),
                "gates": {key: {"passed": True, "evidence": [evidence(p) for p in refs]}
                          for key, refs in gates.items()},
                "scientific_status": "admitted_for_falsifiable_training_not_efficacy",
                "frozen_protocol": {"epochs": 30, "seed": 1999, "server": "server-2",
                                    "allowed_gpus": [0], "post_training_design": evidence(design)},
                "limitations": "All adverse initial and wrong-control learning retained; no benefit/necessity/novelty/human approval. Diagnostic weights never deployed.",
            }
            task = {
                "id": ID, "label": "Y43 all-channel point asymmetry confidence",
                "config": CONFIG, "status": "waiting", "depends_on": [],
                "code_commit": commit, "admission_receipt": "pipeline/admissions/" + ID + ".json",
                "requires_exclusive_ownership": True, "min_epochs": 30,
                "patience": 30, "total_epochs": 30,
            }
            active = [t for entries in queue["gpu_queues"].values() for t in entries
                      if t["status"] in ("running", "waiting", "claiming")]
            assert len(active) == 1 and active[0]["id"] == "AdaRES-Y42"
            assert active[0]["pid"] == 91215 and str(active[0]["gpu"]) == "1"
            assert not any(t["id"] == ID for entries in queue["gpu_queues"].values() for t in entries)
            payload = {"old": old_bundle["files"], "new": bundle["files"],
                       "terminal_sources": terminal["source_hashes"]}
            program = r'''
import hashlib,pathlib,subprocess
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")
cmd=pathlib.Path("/proc/91215/cmdline").read_bytes().split(b"\0")
assert b"configs_song/model_paper5/adares_y42_point_covariance_res.yaml" in cmd
assert cmd[cmd.index(b"--gpu_ids")+1]==b"1"
assert not pathlib.Path("/proc/116906").exists()
for row in P["old"]+P["new"]:
    assert hashlib.sha256((root/row["path"]).read_bytes()).hexdigest()==row["sha256"]
for path,sha in P["terminal_sources"].items():
    assert hashlib.sha256((root/"AdaRES"/path).read_bytes()).hexdigest()==sha
print("Y42_AND_IMMUTABLE_SOURCES_VERIFIED")
'''
            result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                     "songcl@192.168.1.163",
                                     server["python"] + " -c " + shlex.quote("P=" + repr(payload) + "\n" + program)],
                                    capture_output=True, timeout=60)
            assert result.returncode == 0 and result.stdout.decode().strip() == "Y42_AND_IMMUTABLE_SOURCES_VERIFIED"
            original_ssh = monitor.ssh

            def scoped_ssh(host, user, command, *positional, **keywords):
                assert host == server["host"] and user == server["user"]
                if command.startswith("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total"):
                    command = command.replace("nvidia-smi ", "nvidia-smi -i 0 ", 1)
                if "nvidia-smi" in command:
                    assert "nvidia-smi -i 0 " in command
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
                monitor._process_queue(
                    queue, "server-2", server["host"], server["user"], str(ROOT), {"1"}, {"0"},
                    events, max_tasks_per_gpu=1, remote_code_root=server["remote_code_root"],
                    current_tasks=1, max_tasks_total=2)
            finally:
                monitor.ssh = original_ssh
            latest = monitor.load_queue(str(ROOT))
            for server_id, entries in preserved_queue["gpu_queues"].items():
                assert [t for t in latest["gpu_queues"][server_id] if t["id"] != ID] == entries
            selected = next(t for t in latest["gpu_queues"]["server-2"] if t["id"] == ID)
            path = save("y43_dispatch_0422", {
                "complete": True, "observed_at_local": datetime.now().astimezone().isoformat(),
                "task": selected, "events": events, "receipt": evidence(task["admission_receipt"]),
                "preserved_y42_exact": True, "scope": "Canonical GPU0 dispatch, not a repeated preflight."})
            candidate["evidence"].extend([evidence(path), evidence(task["admission_receipt"]),
                                          evidence(args.decision)])
            candidate.update(stage="deployment", scientific_status="admitted_not_verified_training",
                             next_action="Independently verify original Y43 PID/config/GPU0 and increasing native Train steps; never restart blindly.")
            print(json.dumps({"task": selected, "events": events}, indent=2))
        else:
            assert candidate["stage"] == "deployment"
            task = next(t for t in queue["gpu_queues"]["server-2"] if t["id"] == ID)
            assert task["status"] == "running" and str(task["gpu"]) == "0"
            payload = {"task": task, "files": old_bundle["files"] + bundle["files"],
                       "terminal_sources": terminal["source_hashes"]}
            program = r'''
import hashlib,json,pathlib,re,subprocess,time
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")
code=root/"AdaRES"
t=P["task"]
assert (code/".codex_source_commit").read_text().strip()==t["code_commit"]
for row in P["files"]:
    assert hashlib.sha256((root/row["path"]).read_bytes()).hexdigest()==row["sha256"]
for path,sha in P["terminal_sources"].items():
    assert hashlib.sha256((code/path).read_bytes()).hexdigest()==sha
pid=t["pid"]
assert pathlib.Path("/proc/"+str(pid)+"/cwd").resolve()==code
argv=pathlib.Path("/proc/"+str(pid)+"/cmdline").read_bytes().split(b"\0")
assert b"tools/train.py" in argv and t["config"].encode() in argv
assert argv[argv.index(b"--gpu_ids")+1]==b"0"
old=pathlib.Path("/proc/91215/cmdline").read_bytes().split(b"\0")
assert b"configs_song/model_paper5/adares_y42_point_covariance_res.yaml" in old
assert old[old.index(b"--gpu_ids")+1]==b"1"
captures=[]
limit=time.monotonic()+90
while len(captures)<2:
    assert pathlib.Path("/proc/"+str(pid)).exists()
    with open(t["log"],"rb") as stream:
        stream.seek(0,2)
        stream.seek(max(0,stream.tell()-24000))
        text=stream.read().decode("utf-8",errors="replace")
    assert not any(v in text for v in ("Traceback","CUDA out of memory","RuntimeError"))
    steps=[tuple(map(int,m)) for m in re.findall(r"Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)",text)]
    if steps:
        epoch,total,step,totalsteps=steps[-1]
        if not captures or (epoch,step)>(captures[0]["epoch"],captures[0]["step"]):
            captures.append(dict(epoch=epoch,epochs=total,step=step,total_steps=totalsteps))
    if len(captures)<2:
        assert time.monotonic()<limit,"No increasing original native progress"
        time.sleep(5)
assert all(c["epochs"]==30 for c in captures)
assert pathlib.Path("/proc/91215/cmdline").read_bytes().split(b"\0")==old
used,util=map(int,subprocess.check_output(["nvidia-smi","-i","0",
    "--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"]).decode().strip().split(","))
print(json.dumps(dict(complete=True,id="AdaRES-Y43",pid=pid,config=t["config"],log=t["log"],
    code_commit=t["code_commit"],actual_gpu=0,native_steps_increased=True,captures=captures,
    gpu0_device_memory_mib=used,gpu0_device_utilization=util,
    preserved_y42_pid=91215,preserved_y42_gpu=1,source_hashes_verified=True,
    limitations=["Bounded logs, not whole-log crash absence.",
                 "Device occupation is not isolated peak or efficacy."])))
'''
            output = ROOT / (BASE + "y43_launch_0422_" + args.token + ".json")
            assert not output.exists() and not output.with_suffix(".log").exists()
            started = datetime.now().astimezone().isoformat()
            response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                       "songcl@192.168.1.163",
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
                             next_action="Preserve original Y43 GPU0/Y42 GPU1 fixed30. Terminal events require frozen9508/141 twice EACH/FIRST/scenes10000seed1999/size/object/harmvsEither/repeat/trainedcost; no blind restart or retune.")
            print(json.dumps({"path": output.relative_to(ROOT).as_posix(), "proof": proof}, indent=2))
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
finally:
    lock.close()
