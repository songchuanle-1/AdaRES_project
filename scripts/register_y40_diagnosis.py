"""Register Y40 under the exact current foreground and project leases."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "5e172bd47ee3450dbee928bdecaede7f"
owner = json.loads(
    (ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text(
        encoding="utf-8"
    )
)
assert owner["token"] == TOKEN
assert owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    assert not any(row["id"] == "AdaRES-Y40" for row in work["candidates"])
    reference = "pipeline/analyses/Y40_diagnosis_prereg_5e172bd4.md"
    candidate = {
        "id": "AdaRES-Y40",
        "hypothesis_id": "final-mask-word-support-alignment-confidence",
        "stage": "diagnosis",
        "hypothesis": (
            "Direct valid-word to final-mask-supported feature alignment may "
            "distinguish expression/support pairings beyond pooled mask means; "
            "existing decoder encoding and trained utility remain unknown."
        ),
        "matched_control": (
            "Physical W2 and any exact existing word/support readout; fixed "
            "final queries, masks, support-feature/token means, mass and "
            "padding; repeat and joint permutation; no numeric search."
        ),
        "next_action": (
            "Audit actual final-indicator source inputs and execute the CPU "
            "equal-mean word/support intervention on any existing realization. "
            "Persist hashes before rejection or native implementation; "
            "preserve Y39 and perform Y37 frozen terminal evaluation."
        ),
        "evidence": [
            {
                "path": reference,
                "sha256": hashlib.sha256((ROOT / reference).read_bytes()).hexdigest(),
            }
        ],
    }
    work["candidates"].append(candidate)
    path.write_text(
        json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(candidate, indent=2))
finally:
    lock.close()
