"""Attach real launch proof and prospective evaluation to the stable Y39."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
parser.add_argument("--decision", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    queue = monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"]
    task = next(entry for entry in queue if entry["id"] == "AdaRES-Y39")
    assert task["status"] == "running" and task["launch_verified"] is True
    assert str(task["gpu"]) == "1" and task["total_epochs"] == 30
    launch_path = "pipeline/analyses/y39_launch_" + args.token + ".json"
    launch = json.loads((ROOT / launch_path).read_text())
    assert launch["complete"] and launch["native_steps_increased"]
    assert launch["pid"] == task["pid"] and launch["code_commit"] == task["code_commit"]
    old = next(entry for entry in queue if entry["id"] == "AdaRES-Y37")
    assert old["status"] == "running" and old["pid"] == 20348 and str(old["gpu"]) == "0"
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    item = next(candidate for candidate in work["candidates"] if candidate["id"] == "AdaRES-Y39")
    assert item["stage"] == "validation"
    references = ["pipeline/admissions/AdaRES-Y39.json", args.decision, launch_path,
                  "scripts/normalize_y39_committed_bytes.py", "scripts/commit_y39_code.py",
                  "scripts/admit_dispatch_y39.py", "scripts/capture_y39_launch.py",
                  "scripts/finalize_y39_training.py"]
    references += ["pipeline/analyses/y39_" + name + "_" + args.token + ".json"
                   for name in ("source_bundle_lf", "readout_cpu_lf", "extract_cpu_lf", "dispatch", "commit")]
    for relative in references:
        entry = dict(path=relative, sha256=hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())
        prior = next((value for value in item["evidence"] if value["path"] == relative), None)
        assert prior is None or prior == entry, "Never rewrite bound evidence"
        if prior is None:
            item["evidence"].append(entry)
    item.update(
        stage="training", code_commit=task["code_commit"],
        scientific_status="training_not_efficacy",
        next_action="Preserve original fixed30/seed1999 Y39 GPU1 and Y37 PID20348/GPU0. "
                    "Resume only on canonical terminal/crash events; never blindly restart. "
                    "On Y39 terminal execute frozen full9508/141 Y39/W2/B1 twice EACH, FIRST primary, "
                    "whole-scene bootstrap10000 seed1999, TRAINcuts1487/3325 subgroup harm, "
                    "repeat noise and trained paired costs. Negative partition ceiling diagnosis, "
                    "initial adverse losses, wrong-control learnability and sparse continuation noise retained.",
    )
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(stage=item["stage"], launch_pid=task["pid"], commit=task["code_commit"],
                          active=sum(entry["status"] in ("running", "waiting", "claiming") for entry in queue),
                          preserved_y37_pid=old["pid"], evidence_count=len(item["evidence"]))))
finally:
    lock.close()
