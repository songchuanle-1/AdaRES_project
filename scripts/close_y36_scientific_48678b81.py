"""Close the same Y36 after canonical terminal confirmation without re-testing."""

import argparse
import csv
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

lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    queue_path = ROOT / "pipeline/queue.json"
    csv_path = ROOT / "pipeline/experiments.csv"
    preserved = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (queue_path, csv_path)}
    with csv_path.open(encoding="utf-8", newline="") as stream:
        row = next(item for item in csv.DictReader(stream) if item["id"] == "AdaRES-Y36")
    assert row["status"] == "done_complete" and row["best_epoch"] == "30/30"
    prior_token = "1fb792ef616d43b985a0a12024cb9fc8"
    comparison = json.loads((ROOT / ("pipeline/analyses/y36_clustered_comparison_"
                                     + prior_token + ".json")).read_text())
    harm = [(reference, group) for reference in ("versus_w2", "versus_b1")
            for group, value in comparison[reference].items()
            if group != "all" and value.get("frozen_material_harm_ci_upper_below_zero")]
    assert not harm and not comparison["target_reached"]
    for suffix in ("trained_cost", "postaudit"):
        assert json.loads((ROOT / ("pipeline/analyses/y36_" + suffix + "_"
                                  + prior_token + ".json")).read_text())["complete"]
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidate = next(item for item in work["candidates"] if item["id"] == "AdaRES-Y36")
    assert candidate["stage"] == "training"
    reference = "pipeline/analyses/AdaRES-Y36_frozen_review_1fb792ef.md"
    entry = next(item for item in candidate["evidence"] if item["path"] == reference)
    assert entry["sha256"] == hashlib.sha256((ROOT / reference).read_bytes()).hexdigest()
    candidate["stage"] = "complete"
    candidate["next_action"] = (
        "Canonical done_complete/fixed30 confirmed in tick48678b81. Reused all "
        "hashed original frozen evaluations, scene-bootstrap/subgroups/repeats/"
        "trained-cost/postaudit from tick1fb792ef; no repeated GPU work. SAME "
        "Y36 scientific complete_inconclusive; target unmet and W2 superiority "
        "unresolved, retain W2 scientifically. Numeric goal-checker champion "
        "may be Y36 using rounded ledger .5110, not a scientific promotion. "
        "Preserve checkpoints/source/Y37; no restart, tuning or renamed rerun."
    )
    self_ref = "scripts/close_y36_scientific_48678b81.py"
    candidate["evidence"].append(dict(
        path=self_ref, sha256=hashlib.sha256((ROOT / self_ref).read_bytes()).hexdigest()))
    work_path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == sha
               for path, sha in preserved.items())
    print(json.dumps(dict(id=candidate["id"], stage=candidate["stage"],
                         scientific_status="complete_inconclusive",
                         queue_and_csv_unchanged=True, repeated_gpu_tests=0)))
finally:
    lock.close()
