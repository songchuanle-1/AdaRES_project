"""Register a prospective diagnosis without rewriting historical candidates."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "e636ce91a12e45488093b07fa2fc092b"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    assert not any(row["id"] == "AdaRES-Y37" for row in work["candidates"])
    reference = "pipeline/analyses/Y37_diagnosis_prereg_e636ce91.md"
    candidate = dict(
        id="AdaRES-Y37",
        hypothesis_id="final-mask-global-pair-distance-distribution-confidence",
        stage="diagnosis",
        hypothesis="Global final-mask pair-distance distributions may distinguish "
        "supports with equal feature means and coordinate covariance, beyond "
        "existing overlap/context and nearest-neighbor readouts; utility unknown.",
        matched_control="Physical W2 final-indicator-only intervention; equal "
        "mask/feature/coordinate moments, repeat, point permutation, translation, "
        "rotation and wrong-mask/pair controls. Native128/argmax/released "
        "backbone/seed1999/fixed30/auxiliary losses unchanged; no numeric search.",
        next_action="Execute prediction-blind CPU equal-moment/different-global-"
        "pair-distance fixtures and source-contract audit. Freeze the realization "
        "before actual native TRAIN controls/backward/resources; first preserve "
        "and evaluate terminal Y35 on verified GPU0 using its frozen protocol.",
        evidence=[dict(path=reference,
                       sha256=hashlib.sha256((ROOT / reference).read_bytes()).hexdigest())],
    )
    work["candidates"].append(candidate)
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(candidate, indent=2))
finally:
    lock.close()
