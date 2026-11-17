"""Record tested Y40 rejection and independent Y41 diagnosis under writer lock."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "5e172bd47ee3450dbee928bdecaede7f"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor


def evidence(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


cpu_path = "pipeline/analyses/y40_word_overlap_cpu_" + TOKEN + ".json"
cpu = json.loads((ROOT / cpu_path).read_text())
assert cpu["complete"] and cpu["cpu_cases"] == 32
assert cpu["existing_y35_min"] > 0 and cpu["w2_max"] == cpu["repeat_max"] == 0
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text())
    y40 = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y40")
    assert y40["stage"] == "diagnosis"
    y40.update(
        stage="rejected",
        result="Actual Y35 native word/support path already distinguishes the "
               "equal-token-mean intervention across all 32 seeds; reject the "
               "missing-realization premise, not all language hypotheses.",
        next_action="Do not admit this tested duplicate realization. Revisit "
                    "only a different evidence-backed information deficit.",
    )
    y40["evidence"] += [evidence(cpu_path), evidence("scripts/probe_y40_word_overlap_cpu.py")]
    assert not any(row["id"] == "AdaRES-Y41" for row in work["candidates"])
    work["candidates"].append({
        "id": "AdaRES-Y41", "stage": "diagnosis",
        "hypothesis_id": "final-mask-boundary-probability-mass",
        "hypothesis": "Independent normalization of boundary/interior visual "
                      "messages may discard boundary probability mass.",
        "matched_control": "Physical W2/Y36, fixed query/coordinates/mass, "
                           "constant visual features and equal mask histogram; "
                           "block versus alternating probability assignment.",
        "next_action": "Run preregistered actual-class CPU discrimination and "
                       "invariance tests; if separated, test prediction-blind "
                       "native TRAIN incidence and four-tensor backward.",
        "evidence": [evidence("pipeline/analyses/Y41_diagnosis_prereg_5e172bd4.md")],
    })
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"y40": y40["stage"], "y41": "diagnosis"}))
finally:
    lock.close()
