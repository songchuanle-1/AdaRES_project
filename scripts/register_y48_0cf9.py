"""Register one stable native target-exchangeability diagnosis before audit."""

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

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text())
    assert not any(c["id"] == "AdaRES-Y48" for c in work["candidates"])
    prereg = "pipeline/analyses/Y48_exchangeability_prereg_0cf9.md"
    work["candidates"].append(dict(
        id="AdaRES-Y48", stage="diagnosis", hypothesis_id="native-matched-mask-exchangeability",
        hypothesis="Native matched-indicator labels may assign unequal confidence targets to exactly identical output-mask candidates; equivalence-aware ownership targets may avoid winner-identity suppression without selector/ranking changes.",
        matched_control="Actual W2 native indicator and matching-label source, indistinguishable exact-duplicate query/mask loss-gradient fixtures, all32 frozen TRAIN expressions/128query exact bitset incidence, correct/wrong/equal target and permutation controls.",
        next_action="Audit and execute exact native indicator target/loss on duplicate-query controls; measure nonempty exact duplicate incidence in the frozen prediction-blind TRAIN cohort before further native matching-label work.",
        scientific_status="registered_diagnosis_not_admitted",
        pending_gates=["diagnosis", "controls", "native_backward", "resources", "independence", "constraints"],
        evidence=[dict(path=prereg, sha256=hashlib.sha256((ROOT / prereg).read_bytes()).hexdigest())],
        history=[dict(observed_at_local=datetime.now().astimezone().isoformat(),
                      action="stable_diagnosis_registered_before_source_audit")]))
    path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(id="AdaRES-Y48", stage="diagnosis",
                          evidence=work["candidates"][-1]["evidence"])))
finally:
    writer.close()
