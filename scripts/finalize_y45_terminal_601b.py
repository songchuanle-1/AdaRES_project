"""Reconcile only Y45 using verified fixed-30 terminal artifacts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()


def check_lease():
    owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
    assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 90


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


check_lease()
proof_path = ROOT / ("pipeline/analyses/y45_terminal_601b_" + args.token + ".json")
proof = json.loads(proof_path.read_text(encoding="utf-8"))
assert proof["complete"] and proof["id"] == "AdaRES-Y45"
last = proof["checkpoints"][0]
assert last["epoch"] == last["scheduler_epoch"] == 30
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["allowed_gpus"] == [0, 1]
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    queue = monitor.load_queue(str(ROOT))
    task = next(t for t in queue["gpu_queues"]["server-2"] if t["id"] == "AdaRES-Y45")
    assert task["status"] == "running" and task["pid"] == 78001
    assert str(task["gpu"]) == "1" and task["log"] == proof["log"]
    assert task["code_commit"] == "338a3d7a5b21fa463d116e566b482f0733fde9c2"
    assert subprocess.check_output([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "songcl@192.168.1.163", "test ! -d /proc/78001 && hostname"],
        timeout=20).decode().strip() == "dl01-MS-7C60"
    destination = ROOT / "Checkpoints/server_2" / Path(proof["log"]).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    subprocess.run(["scp", "-q", "songcl@192.168.1.163:" + proof["log"], str(destination)],
                   check=True, timeout=60)
    raw = destination.read_bytes()
    assert hashlib.sha256(raw[-64000:]).hexdigest() == proof["bounded_tail_sha256"]
    check_lease()
    original_pull = monitor.pull_log

    def verified_pull(project_dir, host, user, remote_log, cfg_name, state, srv_name=""):
        assert Path(project_dir) == ROOT and host == "192.168.1.163" and user == "songcl"
        assert remote_log == proof["log"] and cfg_name == "adares_y45_metric_decomposition_res"
        assert state == "done_complete" and srv_name == "server-2"
        return str(destination)

    monitor.pull_log = verified_pull
    events = []
    try:
        monitor.finalize_task(
            str(ROOT), queue, "server-2", server["host"], server["user"], task,
            "adares_y45_metric_decomposition_res", "done_complete", last["best_miou"],
            "30/30", "1", proof["log"], events)
    finally:
        monitor.pull_log = original_pull
    experiment = monitor.load_experiment(str(ROOT), "AdaRES-Y45")
    experiment["terminal_evidence"] = ref(proof_path)
    experiment["terminal_checkpoint_metadata"] = proof["checkpoints"]
    monitor.save_experiment(str(ROOT), experiment)
    deathwatch = monitor.load_deathwatch(str(ROOT))
    deathwatch.get("server-2", {}).pop("adares_y45_metric_decomposition_res", None)
    monitor.save_deathwatch(str(ROOT), deathwatch)
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    for candidate in work["candidates"]:
        cid = candidate["id"]
        if cid not in ("AdaRES-Y45", "AdaRES-Y46", "AdaRES-Y48"):
            continue
        candidate.setdefault("history", []).append(dict(
            observed_at=datetime.now().astimezone().isoformat(),
            prior_stage=candidate["stage"], action="Y45_fixed30_terminal_GPU1_opened"))
        if cid == "AdaRES-Y45":
            candidate.update(
                stage="validation", scientific_status="fixed30_complete_frozen_review_pending",
                next_action="Complete SAME frozen full9508/141 twice EACH/FIRST vs W2+B1, scene/subgroup harm and trained cost review; no restart or tuning.")
        elif cid == "AdaRES-Y46":
            candidate.update(stage="validation", scientific_status="native_feasibility_pending")
        else:
            candidate.update(stage="diagnosis", scientific_status="native_joint_capture_pending")
        for key in ("blocker", "blocker_kind", "required_server", "retry_after", "unblock_condition"):
            candidate.pop(key, None)
        candidate["evidence"].append(ref(proof_path))
    work.pop("design_cycle", None)
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    result = dict(
        complete=True, id="AdaRES-Y45", canonical_state="done_complete", epochs="30/30",
        observed_at_local=datetime.now().astimezone().isoformat(),
        archived_log=ref(destination), terminal_evidence=ref(proof_path), events=events,
        scientific_efficacy="unknown_pending_frozen_review",
        newly_executed_training=0, unchanged_other_candidates=True)
    output = ROOT / ("pipeline/analyses/y45_reconciled_601b_" + args.token + ".json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
finally:
    writer.close()
