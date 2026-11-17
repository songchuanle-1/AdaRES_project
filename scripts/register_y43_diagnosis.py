"""Register the prospective Y43 hypothesis without altering existing evidence."""

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
owner = json.loads(
    (ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text(
        encoding="utf-8"
    )
)
assert owner["token"] == args.token
assert owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

reference = "pipeline/analyses/Y43_diagnosis_prereg_0422d0d8.md"
candidate = {
    "id": "AdaRES-Y43",
    "stage": "diagnosis",
    "hypothesis_id": "all-channel-point-distribution-asymmetry",
    "hypothesis": (
        "All-channel within-superpoint third central moments may provide "
        "point-distribution asymmetry absent from existing explicit mean, "
        "variance and full-covariance confidence summaries."
    ),
    "matched_control": (
        "Physical Y39/Y42 actual extraction/readout; within-SP centered "
        "reflection preserves all32 means/full covariance/counts/masks/query/"
        "geometry while reversing third moments.32 fixed seeds, repeat, "
        "permutation and equal-third-moment controls; no GT at inference."
    ),
    "next_action": (
        "Run actual-class CPU extraction/readout reflection controls and "
        "record the numerical repeat/control floor before deciding whether "
        "the missing-realization premise survives. If supported, implement "
        "independent W2-derived all32 third-moment final-indicator path and "
        "execute native TRAIN controls/backward/identity/resources."
    ),
    "evidence": [
        {
            "path": reference,
            "sha256": hashlib.sha256((ROOT / reference).read_bytes()).hexdigest(),
        }
    ],
    "scientific_status": "diagnosis_not_efficacy",
}
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    assert not any(item["id"] == candidate["id"] for item in work["candidates"])
    work["candidates"].append(candidate)
    path.write_text(
        json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(candidate, indent=2))
finally:
    lock.close()
