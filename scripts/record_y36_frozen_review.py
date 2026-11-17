"""Attach real Y36 evidence without accelerating canonical terminal state."""

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
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

references = [
    "pipeline/analyses/Y36_terminal_execution_freeze_" + args.token[:8] + ".md",
    "pipeline/analyses/AdaRES-Y36_frozen_review_" + args.token[:8] + ".md",
]
names = ("terminal", "y36_full_eval", "w2_full_eval", "b1_full_eval",
         "clustered_comparison", "trained_cost", "postaudit")
references += ["pipeline/analyses/y36_" + name + "_" + args.token + ".json"
               for name in names]
references += ["scripts/" + name + ".py" for name in (
    "capture_y36_terminal", "capture_y36_frozen_terminal",
    "analyze_y36_frozen_terminal", "build_y36_terminal_cost",
    "probe_y36_trained_cost_fixed", "probe_y36_postaudit",
    "write_y36_terminal_review", "record_y36_frozen_review",
)]
inputs = {name: json.loads((ROOT / references[2 + index]).read_text())
          for index, name in enumerate(names)}
assert all(inputs[name]["complete"] and inputs[name]["completed_passes"] == 2
           for name in ("y36_full_eval", "w2_full_eval", "b1_full_eval"))
assert inputs["trained_cost"]["complete"] and inputs["postaudit"]["complete"]
assert inputs["postaudit"]["source_bytes_preserved"]
comp = inputs["clustered_comparison"]
harm = [
    dict(reference=reference, group=group, delta=row["paired_miou_difference"],
         ci95=row["scene_clustered_ci95"])
    for reference in ("versus_w2", "versus_b1")
    for group, row in comp[reference].items()
    if group != "all" and row.get("frozen_material_harm_ci_upper_below_zero")
]
status = ("rejected" if harm else
          "target_reached_review_pending" if comp["target_reached"] else
          "complete_inconclusive")
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    queue_path = ROOT / "pipeline/queue.json"
    csv_path = ROOT / "pipeline/experiments.csv"
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (queue_path, csv_path)}
    queue = json.loads(queue_path.read_text())
    original = next(task for task in queue["gpu_queues"]["server-2"]
                    if task["id"] == "AdaRES-Y36")
    assert original["pid"] == 100018 and str(original["gpu"]) == "1"
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidate = next(item for item in work["candidates"] if item["id"] == "AdaRES-Y36")
    assert candidate["stage"] == "training"
    for relative in references:
        entry = dict(path=relative, sha256=hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())
        previous = next((row for row in candidate["evidence"] if row["path"] == relative), None)
        assert previous is None or previous == entry
        if previous is None:
            candidate["evidence"].append(entry)
    candidate["next_action"] = (
        "Original Y36 PID100018 absent; last/best30 and immutable checkpoints verified. "
        "Frozen9508/141 twice candidate/W2/B1 FIRSTprimary, scene-bootstrap10000seed1999/"
        "TRAINcuts1487/3325, repeat-noise, trained-cost and postaudit are complete. "
        "Scientific review: " + status + "; subgroup_harm=" + json.dumps(harm) + ". "
        "Canonical begin remains running/deathwatch1of3; stage training is retained only "
        "pending canonical terminal confirmation. On that event reuse THESE hashed "
        "completed measurements/review and record the SAME ID disposition; do not "
        "repeat GPU validation, restart, tune or rename. No promotion this tick; "
        "W2 retained. Preserve original source/checkpoints and Y37 PID20348/GPU0."
    )
    work_path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == sha
               for path, sha in before.items())
    print(json.dumps(dict(
        id=candidate["id"], stage=candidate["stage"], scientific_status=status,
        evidence_references_attached=len(references), queue_and_csv_unchanged=True,
        canonical_terminal_confirmation="pending",
    )))
finally:
    lock.close()
