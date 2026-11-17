"""Register one prospective candidate under the verified foreground lease."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "8fd5e469dee241da8f21da7c870c0d3f"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
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
    assert not any(c["id"] == "AdaRES-Y36" for c in work["candidates"])
    evidence_path = "pipeline/analyses/Y36_diagnosis_prereg_8fd5e469.md"
    candidate = {
        "id": "AdaRES-Y36",
        "hypothesis_id": "final-mask-local-adjacency-contrast-confidence",
        "stage": "diagnosis",
        "hypothesis": "Local final-mask boundary adjacency supplies feature compatibility beyond global point-mass competition and mask covariance; absence, necessity and efficacy require discrimination.",
        "matched_control": "Physical W2 final-indicator-only path; same masks/native128/argmax/auxiliary losses/released backbone/seed1999/fixed30. Equal feature-mask marginals with different adjacency, joint permutation, repeat, wrong-edge and detach controls; no numeric or validation search.",
        "next_action": "Audit W2/Y31 native coordinate and feature contracts, then execute actual CPU local-edge evidence discrimination, including equal-marginal/different-adjacency and joint permutation controls. Continue native TRAIN controls/backward/resources if evidence supports the realization; complete Y33 frozen paired evaluation first on verified GPU1.",
        "evidence": [{
            "path": evidence_path,
            "sha256": hashlib.sha256((ROOT / evidence_path).read_bytes()).hexdigest(),
        }],
    }
    work["candidates"].append(candidate)
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(candidate, indent=2))
finally:
    lock.close()
