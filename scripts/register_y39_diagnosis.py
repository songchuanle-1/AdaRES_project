"""Close tested Y38 coverage and register an independent point-pooling gap."""

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

lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    previous = next(item for item in work["candidates"] if item["id"] == "AdaRES-Y38")
    assert previous["stage"] == "diagnosis"
    probe = "pipeline/analyses/y38_history_overlap_cpu_" + args.token + ".json"
    result = json.loads((ROOT / probe).read_text())
    assert result["complete"] and result["cpu_cases"] == 32
    assert result["w2_max"] == 0 and result["existing_y30_min"] > 0
    assert result["repeat_max"] == result["permutation_max"] == 0
    for reference in (probe, "scripts/probe_y38_history_overlap_cpu.py"):
        previous["evidence"].append(dict(
            path=reference, sha256=hashlib.sha256((ROOT / reference).read_bytes()).hexdigest()))
    previous["stage"] = "rejected"
    previous["rejection_reason"] = (
        "Actual Y30 already consumes the proposed point-mass mask-history/GRU "
        "information and responds to all32 equal-final-state history controls; "
        "W2 delta0, Y30 minimum .005445152521. Missing-path realization is "
        "covered, not a new independent admission. Not a ban on all history models."
    )
    previous["next_action"] = "Closed SAME Y38 diagnosis; do not rename or re-admit Y30."
    assert not any(item["id"] == "AdaRES-Y39" for item in work["candidates"])
    reference = "pipeline/analyses/Y39_diagnosis_prereg_48678b81.md"
    candidate = dict(
        id="AdaRES-Y39", hypothesis_id="within-superpoint-feature-distribution-confidence",
        stage="diagnosis",
        hypothesis="Native superpoint mean pooling discards within-superpoint "
        "point-feature heterogeneity while masks broadcast over each superpoint. "
        "An explicit all-channel point-distribution confidence path may supply "
        "representability evidence beyond equal superpoint means/counts/geometry.",
        matched_control="Equal SP means/counts/centroids/query/masks with changed "
        "within-SP feature variance; point permutations, native TRAIN exact "
        "union-of-SP IoU ceiling verified by exhaustive small cases; no GT at "
        "inference, no feature subset or numeric search.",
        next_action="Execute physical pooling source audit and CPU equal-mean "
        "point-distribution fixtures/exhaustive ceiling verification; then "
        "first64 distinct native TRAIN superpoint representability diagnosis. "
        "Freeze realization before native controls/backward/resources.",
        evidence=[dict(path=reference,
                       sha256=hashlib.sha256((ROOT / reference).read_bytes()).hexdigest())],
    )
    work["candidates"].append(candidate)
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(y38_stage=previous["stage"], y39=candidate), indent=2))
finally:
    lock.close()
