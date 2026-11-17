"""Persist verified training disposition without replaying admission."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "8fd5e469dee241da8f21da7c870c0d3f"
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
    task = next(t for t in queue if t["id"] == "AdaRES-Y36")
    assert task["status"] == "running" and task["launch_verified"] is True
    assert str(task["gpu"]) == "1" and task["total_epochs"] == 30
    launch_path = "pipeline/analyses/y36_launch_" + TOKEN + ".json"
    launch = json.loads((ROOT / launch_path).read_text())
    assert launch["complete"] and launch["native_steps_increased"]
    assert launch["pid"] == task["pid"] and launch["code_commit"] == task["code_commit"]
    old = next(t for t in queue if t["id"] == "AdaRES-Y35")
    assert old["status"] == "running" and old["pid"] == 108331 and str(old["gpu"]) == "0"
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    item = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y36")
    for relative in (
        "pipeline/analyses/Y36_admission_review_8fd5e469.md",
        "pipeline/analyses/y36_source_bundle_lf_" + TOKEN + ".json",
        "pipeline/analyses/y36_structure_cpu_lf_" + TOKEN + ".json",
        "pipeline/analyses/y36_dispatch_" + TOKEN + ".json",
        launch_path, "pipeline/admissions/AdaRES-Y36.json", args.decision,
        "pipeline/analyses/y36_native_controls_" + TOKEN + ".log",
        "scripts/probe_y36_native_controls.py",
        "scripts/probe_y36_native_controls_v2.py",
        "scripts/probe_y36_physical_native.py",
        "scripts/probe_y36_edge_diagnosis_v2.py",
        "scripts/probe_y36_resources.py",
        "scripts/probe_y36_initial_loss_investigation.py",
        "scripts/normalize_y36_committed_bytes.py",
        "scripts/admit_dispatch_y36.py",
        "scripts/capture_y36_launch.py",
        "scripts/run_adares_authorized_probe.py",
    ):
        entry = evidence(relative)
        prior = next((e for e in item["evidence"] if e["path"] == relative), None)
        assert prior is None or prior == entry, "Never rewrite bound evidence"
        if prior is None:
            item["evidence"].append(entry)
    item["stage"] = "training"
    item["code_commit"] = task["code_commit"]
    item["next_action"] = (
        "Preserve original fixed30/seed1999 Y36 GPU1 native training and Y35 GPU0; "
        "monitor canonical terminal/crash events, never blindly restart. After completion "
        "perform frozen9508/141 Y36/W2/B1 twice first-primary scene-bootstrap10000 "
        "seed1999 TRAINcuts1487/3325 subgroup-harm/repeat-noise/trained-cost evaluation. "
        "Initialization harm/wrong-controls learning/numerical continuation/resource increase "
        "retained; not efficacy/necessity/novelty/Pareto or human approval."
    )
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(stage=item["stage"], launch_pid=task["pid"],
                         commit=task["code_commit"], active=len([t for t in queue if t["status"] in ("running", "waiting", "claiming")]),
                         preserved_y35_pid=old["pid"], evidence_count=len(item["evidence"]))))
finally:
    lock.close()
