"""Persist actual current Y44 evidence without claiming unrun gates."""
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
refs = [
    "pipeline/analyses/y44_pairing_cpu_ff34_" + args.token + ".json",
    "pipeline/analyses/y44_local_bundle_" + args.token + ".json",
    "pipeline/analyses/y44_source_bundle_" + args.token + ".json",
    "pipeline/analyses/y44_parent_cpu_ff34_" + args.token + ".json",
    "pipeline/analyses/Y44_native_freeze_ff347697.md",
    "pipeline/analyses/Y44_upload_encoding_repair_ff347697.md",
    "pipeline/analyses/y44_native_controls_ff34_" + args.token + ".log",
    "pipeline/analyses/Y44_native_sort_adapter_repair_ff347697.md",
    "pipeline/designs/exp_design_Y44_ff347697.md",
    "pipeline/reviews/motivation_review_Y44_ff347697.md",
    "scripts/probe_y44_native_controls_ff34.py",
    "scripts/probe_y44_native_controls_ff34_v2.py",
]
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y44")
    assert candidate["stage"] == "diagnosis"
    candidate.update(stage="validation", scientific_status="controlled_cpu_gap_native_pending",
                     next_action="Execute v2 native TRAIN8/init4 exact pairing/control/backward, then physicalnative/resume/initialloss/resources. No model/gate/tolerance change.")
    for ref in refs:
        assert (ROOT / ref).is_file()
        entry = {"path": ref, "sha256": hashlib.sha256((ROOT / ref).read_bytes()).hexdigest()}
        assert not any(e["path"] == ref for e in candidate["evidence"])
        candidate["evidence"].append(entry)
    path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"id": candidate["id"], "stage": candidate["stage"], "new_evidence": len(refs)}))
finally:
    lock.close()
