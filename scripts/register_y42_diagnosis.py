"""Register one stable diagnosis under the foreground lease and writer lock."""

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
    (ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text()
)
assert owner["token"] == args.token
assert owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor


def evidence(path):
    return {
        "path": path,
        "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
    }


lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text())
    assert not any(row["id"] == "AdaRES-Y42" for row in work["candidates"])
    work["candidates"].append({
        "id": "AdaRES-Y42",
        "stage": "diagnosis",
        "hypothesis_id": "full-channel-point-covariance",
        "hypothesis": "Equal native superpoint means and diagonal point-feature "
                      "variances may conceal cross-channel co-occurrence.",
        "matched_control": "Actual W2/Y39 classes; equal means, diagonal "
                           "variances, counts, geometry, query and masks, "
                           "changed off-diagonal covariance; point permutation "
                           "and repeats; all channels, no subset or tuning.",
        "next_action": "Execute the preregistered actual-class CPU covariance-only "
                       "intervention; then native TRAIN incidence and controls "
                       "if existing paths are blind and invariants hold.",
        "scientific_status": "diagnosis_not_efficacy",
        "evidence": [
            evidence("pipeline/analyses/Y42_diagnosis_prereg_062ae665.md"),
            evidence("scripts/register_y42_diagnosis.py"),
        ],
    })
    path.write_text(
        json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"candidate": "AdaRES-Y42", "stage": "diagnosis"}))
finally:
    lock.close()
