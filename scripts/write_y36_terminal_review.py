"""Review actual frozen Y36 results, retaining canonical deathwatch state."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
names = ("terminal", "y36_full_eval", "w2_full_eval", "b1_full_eval",
         "clustered_comparison", "trained_cost", "postaudit")
paths = {name: ROOT / ("pipeline/analyses/y36_" + name + "_" + args.token + ".json")
         for name in names}
results = {name: json.loads(path.read_text(encoding="utf-8"))
           for name, path in paths.items()}
assert results["trained_cost"]["complete"] and results["postaudit"]["complete"]
assert results["postaudit"]["source_bytes_preserved"]
terminal = results["terminal"]
best = next(row for row in terminal["checkpoints"] if row["path"].endswith("/best.pth"))
last = next(row for row in terminal["checkpoints"] if row["path"].endswith("/last.pth"))
assert last["epoch"] == 30 and best["epoch"] == 30
assert terminal["processes_after"]["original_cmd"] is None
comp = results["clustered_comparison"]
harm = [
    dict(reference=reference, group=group, delta=row["paired_miou_difference"],
         ci95=row["scene_clustered_ci95"])
    for reference in ("versus_w2", "versus_b1")
    for group, row in comp[reference].items()
    if group != "all" and row.get("frozen_material_harm_ci_upper_below_zero")
]
status = ("rejected" if harm else
          "target_reached_review_pending" if comp["target_reached"] else
          "complete_inconclusive")
metrics = []
for name in ("y36", "w2", "b1"):
    capture = results[name + "_full_eval"]
    assert capture["complete"] and capture["completed_passes"] == 2
    assert all(p["summary"]["expressions"] == 9508 and p["summary"]["scenes"] == 141
               for p in capture["passes"])
    primary = capture["passes"][0]["summary"]
    metrics.append(
        "| " + name.upper() + " | " +
        " | ".join(format(primary[key], ".12f")
                   for key in ("miou", "acc25", "acc50", "spmiou")) +
        " | " + format(capture["passes"][1]["summary"]["miou"], ".12f") + " |"
    )
intervals = []
for reference in ("versus_w2", "versus_b1"):
    for group, row in comp[reference].items():
        if "scene_clustered_ci95" not in row:
            continue
        intervals.append(
            "| " + reference + " | " + group + " | " +
            format(row["paired_miou_difference"], ".12f") + " | [" +
            ", ".join(format(value, ".12f") for value in row["scene_clustered_ci95"]) +
            "] | " + str(row["expressions"]) + "/" + str(row["scenes"]) + " | " +
            str(row["frozen_material_harm_ci_upper_below_zero"]) + " |"
        )
reason = (
    "Predeclared material subgroup harm: " + json.dumps(harm) if harm else
    "Target reached numerically; independent harm/Pareto/human review remains pending."
    if comp["target_reached"] else
    "Target unmet; no predeclared subgroup-harm trigger. Intervals crossing zero "
    "remain inconclusive, not equivalence or proof of no harm."
)
evidence = "\n".join(
    "- " + path.relative_to(ROOT).as_posix() + " SHA256 " +
    hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()
)
text = f"""# Experiment Result Analysis - AdaRES_project
Experiment AdaRES-Y36; scientific state {status}.
Observed {datetime.now().astimezone().isoformat()}; no postflight claim.
Canonical begin queue state is running/deathwatch1of3; unchanged by this review.

## 1. Basic Info
Original PID100018/server2GPU1/configadares_y36_local_adjacency_res.
Read-only original process/checkpoint capture {terminal["observed_end_local"]}.
Last/best metadata epoch30 and original PID absent; scheduler terminal confirmation
pending. Actual start time, exit time and duration are not inferred from directory
names and remain unknown. Source6725f9bd6a8f9b47addae6a4b8336d004f73f1ae
preserved under later additive marker67f4a1b2.

## 2. Best Metrics
Best checkpoint epoch{best["epoch"]}; metadata mIoU{best["best_miou"]}.
All9508expressions/141scenes each twice; FIRST pass PRIMARY, no favorable repeat.
| Method | mIoU | Acc25 | Acc50 | spmIoU | Repeated mIoU |
|---|---|---|---|---|---|
{chr(10).join(metrics)}
BestSHA{best["sha256"]}; lastSHA{last["sha256"]}, post-audited unchanged.

## 3. Convergence Summary
Last30/best30 from metadata. Preflight last visible epoch29/bestrounded.5099 is
an earlier monitoring snapshot, not a contradiction or the frozen primary.
Maximum expression repeat change Y36={results["y36_full_eval"]["repeat_max_iou_difference"]},
W2={results["w2_full_eval"]["repeat_max_iou_difference"]},
B1={results["b1_full_eval"]["repeat_max_iou_difference"]}.
Two repetitions quantify empirical noise, not a universal bound or seed variance.

## 4. Comparison
Expression-weighted paired delta, whole-scene bootstrap10000/seed1999;
TRAINrawpointcuts1487/3325. Material subgroup harm if interval upper<0 versus either.
| Reference | Group | Paired delta | Scene CI95 | Expressions/Scenes | Harm CIupper<0 |
|---|---|---|---|---|---|
{chr(10).join(intervals)}
CIcross0 remains inconclusive; official unique/multiple metadata unavailable.
Trained serial cost, original TRAIN8 and largestpair728/6780, two warmups/three
readings; separate native batch2 loss/backward/AdamW stress stays memory-only:
```json
{json.dumps(results["trained_cost"]["summary"], indent=2)}
```
Params {json.dumps(results["trained_cost"]["parameters"])}; native128/oneargmax.
Tiny serial cost does not establish reliable p95, speedup significance or
population lightweight Pareto. Original checkpoints were not resumed or modified.

## 5. Conclusion
{reason}
No promotion this tick; W2 retained. Same stable Y36 realization, no renamed
rerun, epoch selection, threshold/temperature/loss/feature/neighborhood tuning
or blanket family ban. One trained seed is not causal necessity proof.
Y37 original PID20348/GPU0 remains preserved; its final outcome is unknown.
Do not fill GPU1 until canonical capacity opens and real admission gates pass.

## 6. Crash Info
No crash repair or remote restart. Bounded original log contains final30/30
checkpoint saves, no bounded traceback/OOM; this is not whole-log crash absence.
Canonical deathwatch1of3 is not manually accelerated. Missing local exploratory
path scripts/frozen_terminal.py was not executed; the actual frozen evaluator
and existing hash-bound capture protocol were used.

## Evidence
{evidence}
"""
target = ROOT / ("pipeline/analyses/AdaRES-Y36_frozen_review_" + args.token[:8] + ".md")
assert not target.exists()
target.write_text(text, encoding="utf-8", newline="\n")
print(json.dumps(dict(
    path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    scientific_status=status, reason=reason, canonical_lifecycle="running/deathwatch1of3",
)))
