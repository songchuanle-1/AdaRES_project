"""Persist Y37 only after canonical and independent native launch proof."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "e636ce91a12e45488093b07fa2fc092b"
parser = argparse.ArgumentParser()
parser.add_argument("--decision", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor


def evidence(relative):
    path = ROOT / relative
    assert path.is_file()
    return dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    queue = monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"]
    task = next(t for t in queue if t["id"] == "AdaRES-Y37")
    assert task["status"] == "running" and task["launch_verified"] is True
    assert str(task["gpu"]) == "0" and task["total_epochs"] == 30
    launch_path = "pipeline/analyses/y37_launch_v2_" + TOKEN + ".json"
    launch = json.loads((ROOT / launch_path).read_text())
    assert launch["complete"] and launch["native_steps_increased"]
    assert launch["pid"] == task["pid"] and launch["code_commit"] == task["code_commit"]
    old = next(t for t in queue if t["id"] == "AdaRES-Y36")
    assert old["status"] == "running" and old["pid"] == 100018 and str(old["gpu"]) == "1"
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    item = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y37")
    references = [
        "pipeline/analyses/Y37_admission_review_e636ce91.md",
        "pipeline/admissions/AdaRES-Y37.json", args.decision, launch_path,
        "pipeline/analyses/y37_native_controls_v2_" + TOKEN + ".log",
        "pipeline/analyses/y37_launch_" + TOKEN + ".log",
        "scripts/probe_y37_native_controls.py", "scripts/probe_y37_native_controls_v2.py",
        "scripts/probe_y37_native_controls_v3.py", "scripts/fix_y37_gradient_field.py",
        "scripts/probe_y37_physical_native.py", "scripts/probe_y37_resources.py",
        "scripts/probe_y37_initial_loss_investigation.py",
        "scripts/normalize_y37_committed_bytes.py", "scripts/admit_dispatch_y37.py",
        "scripts/capture_y37_launch.py",
        "scripts/capture_y37_launch_v2.py",
    ]
    references += ["pipeline/analyses/y37_" + name + "_" + TOKEN + ".json"
                   for name in ("source_bundle_lf", "structure_cpu_lf", "dispatch")]
    for relative in references:
        entry = evidence(relative)
        prior = next((e for e in item["evidence"] if e["path"] == relative), None)
        assert prior is None or prior == entry, "Never rewrite bound evidence"
        if prior is None:
            item["evidence"].append(entry)
    item["stage"] = "training"
    item["code_commit"] = task["code_commit"]
    item["next_action"] = (
        "Preserve original fixed30/seed1999 Y37 GPU0 and Y36 PID100018/GPU1. "
        "Monitor canonical terminal/crash events, never blindly restart. On Y37 terminal, "
        "run frozen full9508/141 Y37/W2/B1 twice first-primary, scene-bootstrap10000 "
        "seed1999 TRAINcuts1487/3325 subgroup harm/repeat noise/trained paired costs. "
        "Initialization losses, wrong-controls learnability, numerical continuation and "
        "quadratic pair costs retained; no efficacy/necessity/novelty/Pareto assertion."
    )
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(
        stage=item["stage"], launch_pid=task["pid"], commit=task["code_commit"],
        active=sum(t["status"] in ("running", "waiting", "claiming") for t in queue),
        preserved_y36_pid=old["pid"], evidence_count=len(item["evidence"]),
    )))
finally:
    lock.close()
