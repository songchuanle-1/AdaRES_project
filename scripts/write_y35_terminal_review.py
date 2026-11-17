"""Summarize actual frozen Y35 measurements without choosing favorable repeats."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "e636ce91a12e45488093b07fa2fc092b"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
names = ("terminal", "y35_full_eval", "w2_full_eval", "b1_full_eval",
         "clustered_comparison", "trained_cost", "postaudit")
paths = {n: ROOT / ("pipeline/analyses/y35_"+n+"_"+TOKEN+".json") for n in names}
results = {n: json.loads(p.read_text(encoding="utf-8")) for n, p in paths.items()}
assert results["trained_cost"]["complete"] and results["postaudit"]["complete"]
comp = results["clustered_comparison"]
terminal = results["terminal"]
best = next(r for r in terminal["checkpoints"] if r["path"].endswith("/best.pth"))
last = next(r for r in terminal["checkpoints"] if r["path"].endswith("/last.pth"))
assert last["epoch"] == 30 and terminal["processes_after"]["original_cmd"] is None
harm = [(ref, group, row) for ref in ("versus_w2", "versus_b1")
        for group, row in comp[ref].items() if group != "all"
        and row.get("frozen_material_harm_ci_upper_below_zero")]
status = "rejected" if harm else "complete_inconclusive"
metrics = "\n".join(
    "| "+name.upper()+" | "+" | ".join(format(c["passes"][0]["summary"][k], ".12f")
        for k in ("miou", "acc25", "acc50", "spmiou"))+" | "
    +format(c["passes"][1]["summary"]["miou"], ".12f")+" |"
    for name, c in ((n, results[n+"_full_eval"]) for n in ("y35", "w2", "b1"))
)
intervals = "\n".join(
    "| "+ref+" | "+group+" | "+format(row["paired_miou_difference"], ".12f")
    +" | ["+", ".join(format(x, ".12f") for x in row["scene_clustered_ci95"])
    +"] | "+str(row["expressions"])+"/"+str(row["scenes"])
    +" | "+str(row["frozen_material_harm_ci_upper_below_zero"])+" |"
    for ref in ("versus_w2", "versus_b1") for group, row in comp[ref].items()
    if "scene_clustered_ci95" in row
)
evidence = "\n".join("- "+str(p.relative_to(ROOT)).replace("\\", "/")+" SHA256 "
                     +hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values())
reason = ("Frozen subgroup harm: "+json.dumps(
    [dict(reference=ref, group=group, delta=row["paired_miou_difference"],
          ci95=row["scene_clustered_ci95"]) for ref, group, row in harm])
    if harm else "Target unmet; subgroup intervals do not establish material harm. "
    "Intervals crossing zero remain inconclusive, not proven equivalent or harmless.")
text = f"""# Experiment Result Analysis - AdaRES_project
Experiment AdaRES-Y35; canonical state done_complete; scientific state {status}.
Observed {datetime.now().astimezone().isoformat()}; no postflight claim.

## 1. Basic Info
Original PID108331/server2GPU0/configadares_y35_language_evidence_res.
Begin confirmed canonicaldone_complete2026-10-10 11:48:56 local after3absences.
Bounded original process/checkpoint capture {terminal["observed_end_local"]}.
Start time/duration not reconstructed from directory names; unknown here.
Independent source b49b744dc2943a38ab4c3aeb788acc6a2cb90011 preserved.

## 2. Best Metrics
Frozen best checkpoint epoch{best["epoch"]}; metadata mIoU{best["best_miou"]}.
All9508expressions/141scenes, first pass PRIMARY, never choose the better repeat.
| Method | mIoU | Acc25 | Acc50 | spmIoU | Repeated mIoU |
|---|---|---|---|---|---|
{metrics}
BestSHA{best["sha256"]}; lastSHA{last["sha256"]}, unchanged during read.

## 3. Convergence Summary
Last30; best17 in checkpoint metadata. Logrounded monitoring and ledgerrounded
values are not the frozen primary. Two repetitions quantify empirical noise
only; Y35 expression maxrepeat{results["y35_full_eval"]["repeat_max_iou_difference"]}.
No whole-loader determinism or seed variance claim.

## 4. Comparison
Whole-scene paired10000 bootstrap seed1999; TRAINpointcuts1487/3325 unchanged.
| Reference | Group | Paired delta | Scene CI95 | Expressions/Scenes | Harm CIupper<0 |
|---|---|---|---|---|---|
{intervals}
CIcross0 inconclusive, never equivalence/noharm.
Trained paired resources:
```json
{json.dumps(results["trained_cost"]["summary"], indent=2)}
```
Params {json.dumps(results["trained_cost"]["parameters"])}.
Native128/oneargmax. TRAIN8plus frozen largestpair728/6780, serial2warmups/3reads,
separate memory-only train stress. Descriptive timings not reliable p95/
significance or trained lightweight Pareto dominance. Production checkpoints
unchanged by stress; post-audit confirms source/job/checkpoint preservation.

## 5. Conclusion
{reason}
Target.515 unmet; keep W2 champion. Close SAME realization with actual evidence,
no renamed rerun/restart/epoch selection or blanket ban on all word/geometry
routes. The initialization controls did not establish semantic necessity;
one trained seed cannot establish the population mechanism.
Y36 original PID100018/GPU1 preserved and its trained outcome remains unknown.

## 6. Crash Info
No new crash intervention. Bounded terminal tail noTraceback/noOOM is not
whole-log crash absence. First local terminal adapter failed because canonical
completed Y35 had been pruned from active queue; corrected to frozen original
identity without remote restart. No gate/protocol/goal change.

## Evidence
{evidence}
"""
target = ROOT / "pipeline/analyses/AdaRES-Y35_done_complete_e636ce91.md"
assert not target.exists()
target.write_text(text, encoding="utf-8", newline="\n")
print(json.dumps(dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                      scientific_status=status, reason=reason)))
