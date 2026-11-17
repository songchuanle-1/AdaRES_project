"""Register the prospective decoder-history diagnosis under an owned lease."""

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
    assert not any(item["id"] == "AdaRES-Y38" for item in work["candidates"])
    reference = "pipeline/analyses/Y38_diagnosis_prereg_48678b81.md"
    candidate = dict(
        id="AdaRES-Y38", hypothesis_id="explicit-decoder-mask-history-confidence",
        stage="diagnosis",
        hypothesis="Explicit point-mass-weighted soft-mask change history across "
        "native decoder stages may supply query-confidence evidence beyond "
        "final-state readouts. Native utility and independence are unproven.",
        matched_control="Identical final mask/query/features/masses with distinct "
        "intermediate histories; repeat/permutation/wrong-history controls; "
        "physical W2 and targeted historical source audit. Masks, losses, "
        "native128/argmax, fixed30/seed1999 and backbone unchanged.",
        next_action="Execute physical W2 CPU source-contract audit and matched "
        "equal-final-state/different-history fixtures; preserve running Y37. "
        "Then freeze an explicit readout and run native TRAIN incidence and "
        "matched controls if the information gap survives this first test.",
        evidence=[dict(path=reference,
                       sha256=hashlib.sha256((ROOT / reference).read_bytes()).hexdigest())],
    )
    work["candidates"].append(candidate)
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(candidate, indent=2))
finally:
    lock.close()
