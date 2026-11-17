"""Canonical terminal reconciliation of the verified SAME Y44 only."""

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
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
proof_path = ROOT / ("pipeline/analyses/y44_terminal_392a_" + args.token + ".json")
proof = json.loads(proof_path.read_text(encoding="utf-8"))
assert proof["complete"] and proof["id"] == "AdaRES-Y44"
last = proof["checkpoints"][0]
assert last["epoch"] == last["scheduler_epoch"] == 30
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    queue = monitor.load_queue(str(ROOT))
    tasks = [t for t in queue["gpu_queues"]["server-2"] if t["id"] == "AdaRES-Y44"]
    assert len(tasks) == 1 and tasks[0]["status"] == "running"
    task = tasks[0]
    assert task["pid"] == 15520 and str(task["gpu"]) == "1" and task["log"] == proof["log"]
    remote_check = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "songcl@192.168.1.163",
         "test ! -d /proc/15520 && hostname"], timeout=20).decode().strip()
    assert remote_check == "dl01-MS-7C60"
    destination = ROOT / "Checkpoints/server_2" / Path(proof["log"]).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    subprocess.run(["scp", "-q", "songcl@192.168.1.163:" + proof["log"], str(destination)],
                   check=True, timeout=60)
    raw = destination.read_bytes()
    assert hashlib.sha256(raw[-64000:]).hexdigest() == proof["bounded_tail_sha256"]
    original_pull = monitor.pull_log

    def verified_pull(project_dir, host, user, remote_log, cfg_name, state, srv_name=""):
        assert Path(project_dir) == ROOT and host == "192.168.1.163" and user == "songcl"
        assert remote_log == proof["log"] and cfg_name == "adares_y44_coordinate_feature_res"
        assert state == "done_complete" and srv_name == "server-2"
        return str(destination)

    monitor.pull_log = verified_pull
    events = []
    try:
        monitor.finalize_task(str(ROOT), queue, "server-2", "192.168.1.163", "songcl",
                              task, "adares_y44_coordinate_feature_res", "done_complete",
                              last["best_miou"], "30/30", "1", proof["log"], events)
    finally:
        monitor.pull_log = original_pull
    exp = monitor.load_experiment(str(ROOT), "AdaRES-Y44")
    exp["terminal_evidence"] = dict(path=proof_path.relative_to(ROOT).as_posix(),
                                  sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest())
    exp["terminal_checkpoint_metadata"] = proof["checkpoints"]
    monitor.save_experiment(str(ROOT), exp)
    deathwatch = monitor.load_deathwatch(str(ROOT))
    deathwatch.pop("adares_y44_coordinate_feature_res", None)
    monitor.save_deathwatch(str(ROOT), deathwatch)
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    for cid, terminal in (("AdaRES-Y43", "y43_terminal_7a40"), ("AdaRES-Y44", "y44_terminal_392a")):
        candidate = next(c for c in work["candidates"] if c["id"] == cid)
        candidate.setdefault("history", []).append(dict(
            observed_at=datetime.now().astimezone().isoformat(), prior_stage=candidate["stage"],
            action="original_training_finished_resume_frozen_validation"))
        candidate.update(stage="validation", scientific_status="fixed30_complete_frozen_review_pending",
                         next_action="Finish the SAME frozen full9508/141 twice EACH/FIRST vs W2/B1, scene/subgroup harm and trained paired-cost review; no restart or retuning.")
        for key in ("blocker", "blocker_kind", "required_server", "retry_after", "unblock_condition"):
            candidate.pop(key, None)
        path = ROOT / ("pipeline/analyses/" + terminal + "_" + args.token + ".json")
        candidate["evidence"].append(dict(path=path.relative_to(ROOT).as_posix(),
                                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    y45 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y45")
    y45.setdefault("history", []).append(dict(
        observed_at=datetime.now().astimezone().isoformat(), prior_stage=y45["stage"],
        action="GPU1_naturally_free_native_diagnosis_executable"))
    y45.update(stage="diagnosis", scientific_status="native_target_capture_pending")
    for key in ("blocker", "blocker_kind", "required_server", "retry_after", "unblock_condition"):
        y45.pop(key, None)
    work.pop("design_cycle", None)
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    output = ROOT / ("pipeline/analyses/y44_reconciled_392a_" + args.token + ".json")
    assert not output.exists()
    result = dict(complete=True, id="AdaRES-Y44", canonical_state="done_complete",
                  observed_at_local=datetime.now().astimezone().isoformat(),
                  epochs="30/30", archived_log=dict(path=destination.relative_to(ROOT).as_posix(),
                                                   sha256=hashlib.sha256(raw).hexdigest()),
                  terminal_evidence=exp["terminal_evidence"], events=events,
                  preserved_interruption_history=True, scientific_efficacy="unknown_pending_frozen_review")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
finally:
    writer.close()
