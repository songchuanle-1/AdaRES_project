"""Register the stable joint-association diagnosis under the owned lease."""

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


def evidence(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text())
    assert not any(row["id"] == "AdaRES-Y44" for row in work["candidates"])
    work["candidates"].append({
        "id": "AdaRES-Y44",
        "stage": "diagnosis",
        "hypothesis_id": "joint-coordinate-feature-association",
        "hypothesis": "Complete point-feature marginal moments discard coordinate-feature pairing at fixed post-backbone features.",
        "matched_control": "Actual W2/Y39/Y42/Y43 classes; fixed coordinates, labels/counts, queries/masks/padding, and complete feature multisets; within-superpoint pairing permutation; all three coordinates/all channels, nonnegative arms, repeats and joint-order/translation controls.",
        "next_action": "Run preregistered actual-class CPU pairing witness, then inspect exact native coordinate mapping and execute prediction-blind native incidence/controls; never infer training utility from synthetic invariance.",
        "scientific_status": "diagnosis_not_efficacy",
        "evidence": [
            evidence("pipeline/analyses/Y44_diagnosis_prereg_ff347697.md"),
            evidence("scripts/register_y44_diagnosis_ff347697.py"),
        ],
    })
    path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"id": "AdaRES-Y44", "stage": "diagnosis"}))
finally:
    lock.close()
