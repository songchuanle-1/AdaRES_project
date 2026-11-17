"""Set the same Y35 disposition using completed actual frozen evidence."""

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
    comp = json.loads((ROOT / ("pipeline/analyses/y35_clustered_comparison_"+TOKEN+".json")).read_text())
    harm = [(ref, group, row) for ref in ("versus_w2", "versus_b1")
            for group, row in comp[ref].items() if group != "all"
            and row.get("frozen_material_harm_ci_upper_below_zero")]
    for name in ("trained_cost", "postaudit"):
        assert json.loads((ROOT / ("pipeline/analyses/y35_"+name+"_"+TOKEN+".json")).read_text())["complete"]
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y35")
    references = ["pipeline/analyses/AdaRES-Y35_done_complete_e636ce91.md"]
    references += ["pipeline/analyses/y35_"+n+"_"+TOKEN+".json" for n in (
        "terminal", "y35_full_eval", "w2_full_eval", "b1_full_eval",
        "clustered_comparison", "trained_cost", "postaudit")]
    for relative in references:
        entry = dict(path=relative, sha256=hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())
        prior = next((e for e in candidate["evidence"] if e["path"] == relative), None)
        assert prior is None or prior == entry
        if prior is None:
            candidate["evidence"].append(entry)
    candidate["stage"] = "rejected" if harm else "complete"
    if harm:
        candidate["rejection_reason"] = "Frozen subgroup CIupper<0: "+json.dumps([
            dict(reference=ref, group=group, delta=row["paired_miou_difference"],
                 ci95=row["scene_clustered_ci95"]) for ref, group, row in harm])
    candidate["next_action"] = (
        "Closed SAME Y35 fixed30 realization after frozen full9508/141 twice/"
        "freshW2B1 first-primary/groupCI10000seed1999/TRAINcuts1487/3325/"
        "trainedpairedcost/checkpointaudit. Target unmet; retain W2. "
        + ("Predeclared subgroup harm observed; no renamed rerun. " if harm else
           "No promotion; CIcross0 remains inconclusive, not equivalence/noharm. ")
        + "Preserve original source/checkpoints; no blind restart, epoch or protocol tuning."
    )
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True)+"\n", encoding="utf-8")
    print(json.dumps(dict(id=candidate["id"], stage=candidate["stage"], harm_count=len(harm))))
finally:
    lock.close()
