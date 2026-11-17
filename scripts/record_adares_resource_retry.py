"""Observe existing training and retry due frozen diagnostics without dispatch."""

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

FROZEN = {
    "AdaRES-Y46": (
        "scripts/probe_y46_native_gate_3c6a.py",
        "pipeline/analyses/y46_native_gate_freeze_3c6a_3c6a88e6bd334acb8889d557171b1ce6.json",
    ),
    "AdaRES-Y48": (
        "scripts/probe_y48_joint_native_0cf9.py",
        "pipeline/analyses/y48_joint_native_freeze_0cf9_0cf9bfefb82447fcb2abf284a77480e2.json",
    ),
}
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
parser.add_argument("--ids", nargs="+", choices=sorted(FROZEN), required=True)
args = parser.parse_args()
ids = list(dict.fromkeys(args.ids))
owner_path = ROOT / "pipeline/headless/.foreground.lockdir/owner.json"


def check_lease(reserve):
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    assert owner["token"] == args.token
    assert owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert (
        datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)
    ).total_seconds() > reserve


def evidence(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


def save(path, value):
    check_lease(30)
    with (ROOT / path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    return evidence(path)


check_lease(150)
# Reuse the bounded PID/config/source/native-step observer without running its recorder.
nodes = [
    node for node in ast.walk(
        ast.parse((ROOT / "scripts/record_adares_3c6a.py").read_text(encoding="utf-8"))
    )
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "program" for target in node.targets)
]
assert len(nodes) == 1 and isinstance(nodes[0].value, ast.Constant)
program = ast.literal_eval(nodes[0].value)
assert 'root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")' in program
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["allowed_gpus"] == [0, 1]
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None, "Pipeline writer busy; no project mutations"
try:
    check_lease(150)
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidates = {candidate["id"]: candidate for candidate in work["candidates"]}
    frozen = {}
    for candidate_id in ids:
        candidate = candidates[candidate_id]
        assert candidate["stage"] == "blocked"
        assert datetime.fromisoformat(candidate["retry_after"]) <= datetime.now(timezone.utc)
        refs = {row["path"]: row["sha256"] for row in candidate["evidence"]}
        frozen[candidate_id] = [evidence(path) for path in FROZEN[candidate_id]]
        assert all(row["sha256"] == refs[row["path"]] for row in frozen[candidate_id])
    untouched_before = {
        key: json.dumps(value, sort_keys=True)
        for key, value in candidates.items()
        if key not in ids and key != "AdaRES-Y45"
    }
    task = next(
        item for item in monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"]
        if item["id"] == "AdaRES-Y45"
    )
    assert task["status"] == "running" and task["launch_verified"]
    assert int(task["gpu"]) == 1 and task["pid"] == 78001
    assert task["code_commit"] == "338a3d7a5b21fa463d116e566b482f0733fde9c2"
    bundle = json.loads(
        (ROOT / "pipeline/analyses/y45_source_bundle_392a_392a5a9f5fe242d7a3109a50e6060fda.json")
        .read_text(encoding="utf-8")
    )
    started = datetime.now().astimezone().isoformat()
    response = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "songcl@192.168.1.163",
            server["python"] + " -c "
            + shlex.quote("P=" + repr({"task": task, "files": bundle["files"]}) + "\n" + program),
        ],
        capture_output=True, timeout=120,
    )
    check_lease(30)
    stem = "pipeline/analyses/y45_continuation_resource_" + args.token
    with (ROOT / (stem + ".log")).open("xb") as stream:
        stream.write(response.stdout + response.stderr)
    assert response.returncode == 0, "Observation failed; raw evidence retained; no blind restart"
    observed = json.loads(response.stdout)
    observed.update(
        observed_start_local=started,
        observed_end_local=datetime.now().astimezone().isoformat(),
    )
    live_ref = save(stem + ".json", observed)
    assert observed["native_steps_increased"], "No fresh native progress; reconcile without relaunch"
    candidates["AdaRES-Y45"]["evidence"].append(live_ref)
    busy = all(row["pids"] for row in observed["GPUs"])
    idle = [
        row["gpu"] for row in observed["GPUs"]
        if not row["pids"] and row["utilization"] < 10 and row["memory_MiB"] < 1024
    ]
    retries = {}
    if busy:
        assert task["pid"] not in observed["GPUs"][0]["pids"]
        retry = (datetime.now().astimezone() + timedelta(minutes=30)).isoformat()
        disposition_ref = save(
            "pipeline/analyses/candidate_resource_retry_" + args.token + ".json",
            {
                "complete": True, "candidate_ids": ids, "observed": live_ref,
                "unchanged_frozen_artifacts": frozen,
                "actual_test": "Live GPU0/1 occupation and exact Y45 PID/config/cwd/seven source hashes with increasing native steps.",
                "new_native_diagnostics_executed": 0,
                "repeated_CPU_or_source_controls": False,
                "scientific_status_unchanged": {
                    key: candidates[key]["scientific_status"] for key in ids
                },
                "pending_gates_unchanged": {
                    key: candidates[key]["pending_gates"] for key in ids
                },
                "actual_native_mixed_label_groups": None,
                "retry_after": retry,
                "limitations": "Fresh resource evidence only, not scientific validation, route exhaustion, or admission.",
            },
        )
        for candidate_id in ids:
            candidate = candidates[candidate_id]
            candidate.update(
                blocker="Fresh live occupation: external compute PID on server2 GPU0 and original Y45 on GPU1. SAME frozen native diagnostic needs an idle exclusive CUDA slot; useful non-GPU controls already executed.",
                blocker_kind="live_external_gpu_occupation",
                required_server="server-2",
                retry_after=retry,
                unblock_condition="Live-idle authorized server2 GPU0/1 with no compute PID and an exclusive diagnostic lease; preserve every existing job.",
            )
            candidate["evidence"].extend([live_ref, disposition_ref])
            retries[candidate_id] = retry
        work["design_cycle"] = {
            "stop_reason": "external_resource_blocked",
            "attempted_ids": ids,
            "evidence": [live_ref, disposition_ref],
            "resume_condition": "A genuinely idle authorized GPU with exclusive diagnostic lease opens SAME frozen Y46 native loss/backward/paired-cost or SAME frozen Y48 joint mask/matching capture. Stable IDs, frozen protocols and pending gates remain; do not repeat completed CPU/source reviews.",
        }
    else:
        for candidate_id in ids:
            candidates[candidate_id]["evidence"].append(live_ref)
    assert all(json.dumps(candidates[key], sort_keys=True) == value for key, value in untouched_before.items())
    check_lease(30)
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "complete": True,
        "result": "RESOURCE_BLOCKED" if busy else "RESOURCE_RECHECK_REQUIRED",
        "observed": observed, "due_ids": ids,
        "live_idle_GPU_ids": idle, "retry_after": retries, "evidence": live_ref,
    }, indent=2))
finally:
    writer.close()
