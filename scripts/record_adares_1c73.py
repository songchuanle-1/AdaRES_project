"""Record fresh continuation and resource evidence for the existing Y48 retry."""

import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner_path = ROOT / "pipeline/headless/.foreground.lockdir/owner.json"


def check_lease(reserve):
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    assert owner["token"] == args.token
    assert owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert (
        datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)
    ).total_seconds() > reserve


def evidence(path):
    return {
        "path": path,
        "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
    }


def save(path, value):
    check_lease(30)
    with (ROOT / path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    return evidence(path)


check_lease(150)
source = "scripts/probe_y48_joint_native_0cf9.py"
freeze = "pipeline/analyses/y48_joint_native_freeze_0cf9_0cf9bfefb82447fcb2abf284a77480e2.json"
assert evidence(source)["sha256"] == "df3bfa1c2f84526fb046a0506d2bc2277b2fe242a7fe0701bb686794cfaaec62"
assert evidence(freeze)["sha256"] == "bf161c889fccf59fdd1f16f5da9df9890c0064d753141ade4ddf50e3d1280365"
nodes = [
    node
    for node in ast.walk(ast.parse((ROOT / "scripts/record_adares_3c6a.py").read_text()))
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "program" for target in node.targets)
]
assert len(nodes) == 1 and isinstance(nodes[0].value, ast.Constant)
program = ast.literal_eval(nodes[0].value)
assert isinstance(program, str)
assert 'root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")' in program
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["allowed_gpus"] == [0, 1]
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None, "Pipeline writer busy; no project artifact mutations"
try:
    check_lease(150)
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidates = {candidate["id"]: candidate for candidate in work["candidates"]}
    candidate = candidates["AdaRES-Y48"]
    assert candidate["stage"] == "blocked"
    assert datetime.fromisoformat(candidate["retry_after"]) <= datetime.now(timezone.utc)
    y46_before = json.dumps(candidates["AdaRES-Y46"], sort_keys=True)
    task = next(
        task
        for task in monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"]
        if task["id"] == "AdaRES-Y45"
    )
    assert task["status"] == "running" and task["launch_verified"]
    assert int(task["gpu"]) == 1 and task["pid"] == 78001
    assert task["code_commit"] == "338a3d7a5b21fa463d116e566b482f0733fde9c2"
    bundle = json.loads(
        (ROOT / "pipeline/analyses/y45_source_bundle_392a_392a5a9f5fe242d7a3109a50e6060fda.json").read_text()
    )
    started = datetime.now().astimezone().isoformat()
    response = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "songcl@192.168.1.163",
            server["python"] + " -c "
            + shlex.quote("P=" + repr({"task": task, "files": bundle["files"]}) + "\n" + program),
        ],
        capture_output=True,
        timeout=120,
    )
    check_lease(30)
    raw = "pipeline/analyses/y45_continuation_1c73_" + args.token + ".log"
    with (ROOT / raw).open("xb") as stream:
        stream.write(response.stdout + response.stderr)
    assert response.returncode == 0, "Observation failed; raw evidence retained; no blind restart"
    observed = json.loads(response.stdout)
    observed.update(
        observed_start_local=started,
        observed_end_local=datetime.now().astimezone().isoformat(),
    )
    live_ref = save("pipeline/analyses/y45_continuation_1c73_" + args.token + ".json", observed)
    assert observed["native_steps_increased"], "No fresh training progression; reconcile without relaunch"
    candidates["AdaRES-Y45"]["evidence"].append(live_ref)
    busy = all(row["pids"] for row in observed["GPUs"])
    if busy:
        assert task["pid"] not in observed["GPUs"][0]["pids"]
        retry = (datetime.now().astimezone() + timedelta(minutes=30)).isoformat()
        disposition_ref = save(
            "pipeline/analyses/y48_resource_retry_1c73_" + args.token + ".json",
            {
                "complete": True,
                "id": "AdaRES-Y48",
                "observed": live_ref,
                "frozen_native_source": evidence(source),
                "unchanged_freeze": evidence(freeze),
                "actual_test_this_tick": "Live two-card occupation and exact Y45 source/config/PID/cwd with increasing native training steps.",
                "native_joint_capture_executed": False,
                "actual_native_mixed_label_groups": None,
                "all_non_GPU_work_already_executed": True,
                "repeated_CPU_or_source_tests": False,
                "scientific_result": "Unchanged partial diagnosis; native mixed-label incidence still unknown.",
                "retry_after": retry,
                "limitations": "Resource observation is not scientific validation or training admission; freeze and gates unchanged.",
            },
        )
        candidate.update(
            blocker="Fresh live resource observation: external compute PID on server2 GPU0; original Y45 compute PID on GPU1. SAME frozen joint native mask/matching capture requires an idle exclusive CUDA slot; useful non-GPU controls already executed.",
            blocker_kind="live_external_gpu_occupation",
            required_server="server-2",
            retry_after=retry,
            unblock_condition="Observed live-idle server2 GPU0/1 with no compute PID and an exclusive diagnostic lease; preserve all existing jobs.",
        )
        candidate["evidence"].extend([live_ref, disposition_ref])
        work["design_cycle"] = {
            "stop_reason": "external_resource_blocked",
            "attempted_ids": ["AdaRES-Y48"],
            "evidence": [live_ref, disposition_ref],
            "resume_condition": "A genuinely idle authorized GPU with exclusive diagnostic lease opens SAME frozen Y48 joint native capture and SAME Y46 native gate. Keep stable IDs, freezes and gates; do not repeat completed CPU/source reviews.",
        }
    else:
        candidate["evidence"].append(live_ref)
        retry = candidate["retry_after"]
    assert json.dumps(candidates["AdaRES-Y46"], sort_keys=True) == y46_before
    check_lease(30)
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "complete": True,
        "result": "RESOURCE_BLOCKED" if busy else "GPU_OPEN_REQUIRES_FROZEN_PROBE",
        "observed": observed,
        "Y48": candidate["stage"],
        "retry_after": retry,
        "Y46": "unchanged",
        "evidence": live_ref,
    }, indent=2))
finally:
    writer.close()
