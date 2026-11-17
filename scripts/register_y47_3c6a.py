"""Create the stable explicit-between-candidate diagnosis before route audit."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text())
    assert not any(c["id"] == "AdaRES-Y47" for c in work["candidates"])
    prereg = "pipeline/analyses/Y47_overlap_prereg_3c6a.md"
    work["candidates"].append(dict(
        id="AdaRES-Y47", stage="diagnosis",
        hypothesis_id="explicit-pairwise-mask-candidate-context",
        hypothesis="Explicit point-mass pairwise overlap between native candidate masks may expose confidence correspondence information absent from W2 marginal competition summaries.",
        matched_control="Same actual W2 query tensors, equal row mask mass and equal per-point coverage, differing pairwise overlaps; actual closed-route source audit then all32 cached TRAIN expressions/repeats/permutations/wrong correspondence.",
        next_action="Audit actual W2 indicator and execute balanced128-candidate equal-marginal/different-pairwise-overlap CPU witness, then test duplication against relevant closed implementations.",
        scientific_status="registered_diagnosis_not_admitted",
        pending_gates=["diagnosis", "controls", "native_backward", "resources", "independence", "constraints"],
        evidence=[dict(path=prereg, sha256=hashlib.sha256((ROOT / prereg).read_bytes()).hexdigest())],
        history=[dict(observed_at_local=datetime.now().astimezone().isoformat(),
                      action="stable_diagnosis_registered_before_route_audit")]))
    path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(id="AdaRES-Y47", stage="diagnosis",
                          evidence=work["candidates"][-1]["evidence"])))
finally:
    writer.close()
