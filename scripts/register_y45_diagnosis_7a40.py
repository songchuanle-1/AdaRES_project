"""Register one prospective diagnosis, never an admission or queue entry."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text(encoding="utf-8"))
    assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        path = ROOT / "pipeline/research_work.json"
        work = json.loads(path.read_text(encoding="utf-8"))
        assert not any(c["id"] == "AdaRES-Y45" for c in work["candidates"])
        evidence = []
        for name in ("pipeline/analyses/Y45_diagnosis_prereg_7a40f387.md",
                     "scripts/register_y45_diagnosis_7a40.py"):
            evidence.append(dict(path=name, sha256=hashlib.sha256((ROOT / name).read_bytes()).hexdigest()))
        work["candidates"].append(dict(
            id="AdaRES-Y45", stage="diagnosis",
            hypothesis_id="metric-confusion-decomposition",
            hypothesis="Scalar query quality targets may omit distinct false-positive/false-negative point-mass errors.",
            matched_control="Actual W2 scalar-quality/indicator targets at fixed masks/counts; relevant original B2/G1/C1/Y29 target objectives; equal-IoU different-FP/FN witnesses and expression-permuted native targets.",
            next_action="Execute actual-source CPU sufficiency/duplication screen; then native TRAIN point-mass capture with frozen masks and correct/wrong expression target controls. Never admit from synthetic evidence.",
            scientific_status="diagnosis_not_efficacy", evidence=evidence))
        path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(id="AdaRES-Y45", stage="diagnosis", admission=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    main()
