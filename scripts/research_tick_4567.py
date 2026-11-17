"""Persist one foreground diagnostic under its verified project lease."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "4567f5a09ecb4aa89df8f0c2932652df"
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import foreground_tick as tick
import pipeline_monitor as monitor


def lease():
    owner = tick.load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")
    assert owner["token"] == TOKEN
    assert owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert (datetime.fromisoformat(owner["deadline_at"]) -
            datetime.now(timezone.utc)).total_seconds() > 120
    return owner


def ref(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


if __name__ == "__main__":
    owner = lease()
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None, "Project writer busy"
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate_id = "AdaRES-Y52"
        assert not any(c["id"] == candidate_id for c in work["candidates"])
        path = "pipeline/analyses/y52_mass_target_prereg_4567_" + TOKEN + ".json"
        assert not (ROOT / path).exists()
        prereg = {
            "complete": True,
            "id": candidate_id,
            "created_at_local": datetime.now().astimezone().isoformat(),
            "token": TOKEN,
            "hypothesis_id": "native_quality_target_point_mass_aliasing",
            "hypothesis": (
                "A confidence target computed over equal-weight superpoints may "
                "alias masks with different point-level IoU when superpoint "
                "occupancies differ. Determine whether the ACTUAL unchanged W2 "
                "quality target has this mismatch before proposing any correction."
            ),
            "matched_control": (
                "Freeze identical superpoint predictions and object labels, vary "
                "only integer point masses, and compare native quality targets "
                "against exact expanded-point IoU. Include uniform masses, a "
                "nonuniform target-order reversal, empty masks, and a direct "
                "source audit of the real training target and evaluation path."
            ),
            "prediction_and_falsification": (
                "The premise is falsified for the current implementation if native "
                "targets already use the same point masses as point evaluation. "
                "No-op or uniform-mass agreement alone cannot establish it."
            ),
            "sampling": "Source-level exact controls; no validation-based selection.",
            "first_test": "Locate and execute the actual native target helper on CPU.",
            "limits": (
                "A diagnostic only: no model implementation, admission, Git commit "
                "or launch. Cannot infer population efficacy from synthetic controls."
            ),
        }
        tick.save(ROOT / path, prereg)
        work["candidates"].append({
            "id": candidate_id,
            "stage": "diagnosis",
            "hypothesis": prereg["hypothesis"],
            "hypothesis_id": prereg["hypothesis_id"],
            "matched_control": prereg["matched_control"],
            "next_action": prereg["first_test"],
            "evidence": [ref(path)],
        })
        work.pop("design_cycle", None)
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps({"registered": candidate_id, "stage": "diagnosis",
                          "preregistration": ref(path), "token": TOKEN}, indent=2))
    finally:
        lock.close()
