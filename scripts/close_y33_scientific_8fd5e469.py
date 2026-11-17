"""Close the evaluated stable realization with its actual frozen harm result."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
token = "8fd5e469dee241da8f21da7c870c0d3f"
owner = json.loads((root / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

lock = monitor.acquire_lock(str(root))
assert lock is not None
try:
    path = root / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y33")
    comparison = json.loads((root / ("pipeline/analyses/y33_clustered_comparison_" + token + ".json")).read_text())
    assert comparison["versus_w2"]["small"]["frozen_material_harm_ci_upper_below_zero"]
    for name in ("trained_cost", "postaudit"):
        value = json.loads((root / ("pipeline/analyses/y33_" + name + "_" + token + ".json")).read_text())
        assert value["complete"]
    references = [
        "pipeline/analyses/AdaRES-Y33_done_complete_8fd5e469.md",
        "scripts/capture_y33_terminal.py", "scripts/capture_y33_frozen_terminal.py",
        "scripts/analyze_y33_frozen_terminal.py", "scripts/probe_y33_trained_cost.py",
        "scripts/probe_y33_postaudit.py",
    ]
    references += ["pipeline/analyses/y33_" + name + "_" + token + ".json"
                   for name in ("terminal", "y33_full_eval", "w2_full_eval", "b1_full_eval",
                                "clustered_comparison", "trained_cost", "postaudit")]
    for relative in references:
        entry = dict(path=relative, sha256=hashlib.sha256((root / relative).read_bytes()).hexdigest())
        previous = next((e for e in candidate["evidence"] if e["path"] == relative), None)
        assert previous is None or previous == entry
        if previous is None:
            candidate["evidence"].append(entry)
    candidate["stage"] = "rejected"
    candidate["rejection_reason"] = "Frozen small-target material harm versus W2: delta-.024630810599 CI[-.041461090329,-.008034531270]. Original protocol, groups, checkpoints and repeats unchanged."
    candidate["next_action"] = "Closed fixed Y33 realization after all frozen full/paired/noise/cost/release measurements. Target unmet; W2 retained. No renamed rerun. Overall CIcross0 inconclusive, small/W2 CIupper<0 confirms preregistered harm; original checkpoints and source hashes preserved."
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(id=candidate["id"], stage=candidate["stage"],
                         reason=candidate["rejection_reason"])))
finally:
    lock.close()
