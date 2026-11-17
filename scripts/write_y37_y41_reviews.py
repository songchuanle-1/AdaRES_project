"""Persist actual frozen terminal and admission-support reviews under this lease."""

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
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor


def load(name):
    path = ROOT / ("pipeline/analyses/" + name + "_" + TOKEN + ".json")
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["complete"]
    return path.relative_to(ROOT).as_posix(), value


def evidence(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


def save(path, text):
    target = ROOT / path
    assert not target.exists()
    target.write_text(text, encoding="utf-8", newline="\n")
    return evidence(path)


terminal_path, terminal = load("y37_terminal_current")
full_path, full = load("y37_frozen_full_retry")
comparison_path, comparison = load("y37_clustered_comparison")
cost_path, cost = load("y37_trained_cost")
best = next(c for c in terminal["checkpoints"] if c["path"].endswith("/best.pth"))
last = next(c for c in terminal["checkpoints"] if c["path"].endswith("/last.pth"))
assert last["epoch"] == 30 and best["epoch"] == 19
assert terminal["processes_after"]["original_cmd"] is None
harm = [dict(reference=reference, group=group, row=row)
        for reference in ("versus_w2", "versus_b1")
        for group, row in comparison[reference].items()
        if group != "all" and row.get("frozen_material_harm_ci_upper_below_zero")]
assert not comparison["target_reached"]
status = "rejected" if harm else "complete_inconclusive"
metrics = "\n".join(
    "| " + name.upper() + " | " + " | ".join(
        f"{capture['passes'][0]['summary'][key]:.12f}"
        for key in ("miou", "acc25", "acc50", "spmiou")
    ) + f" | {capture['passes'][1]['summary']['miou']:.12f} |"
    for name, capture in full["rows"].items())
intervals = "\n".join(
    f"| {reference} | {group} | {row['paired_miou_difference']:.12f} | "
    f"{row['scene_clustered_ci95']} | {row['expressions']}/{row['scenes']} | "
    f"{row['frozen_material_harm_ci_upper_below_zero']} |"
    for reference in ("versus_w2", "versus_b1")
    for group, row in comparison[reference].items())
refs = [terminal_path, full_path, comparison_path, cost_path]
y37_review = save("pipeline/analyses/AdaRES-Y37_frozen_review_5e172bd4.md", f"""# Experiment Result Analysis - AdaRES_project
Scientific status {status}; canonical lifecycle done_complete confirmed by begin.
Observed {datetime.now().astimezone().isoformat()}; no postflight claim.

## 1. Basic Info
Original server2/GPU0/PID20348/configadares_y37_pair_distribution_res.
Original PID absent and same-config train processes absent in direct capture.
Actual start/exit/duration remain unknown; directory names are not timestamps.
Y39 originalPID102267/GPU1/config and sources were preserved.

## 2. Best Metrics
Original last epoch30; best checkpoint epoch19, metadata mIoU{best['best_miou']}.
Each candidate/W2/B1 full9508expressions/141scenes twice; FIRST pass primary.
| Method | mIoU | Acc25 | Acc50 | spmIoU | Repeated mIoU |
|---|---|---|---|---|---|
{metrics}
BestSHA{best['sha256']}; lastSHA{last['sha256']}.

## 3. Convergence Summary
Last30 and savedbest19 are checkpoint facts, not a full convergence curve.
Rounded automatic CSV .5050 and monitor epochtext30 do not replace metadata.
Maximum expression repeat difference Y37={full['rows']['y37']['repeat_max_iou_difference']},
W2={full['rows']['w2']['repeat_max_iou_difference']},
B1={full['rows']['b1']['repeat_max_iou_difference']}. Near-identical group
means do not prove per-expression determinism. Two repeats are empirical noise,
not universal noise bounds or training-seed uncertainty.

## 4. Comparison
FIRST paired expression-weighted delta, whole-scene bootstrap10000/seed1999.
TRAIN raw-point size cuts1487/3325; subgroup harm if CIupper<0 versus either.
| Reference | Group | Paired delta | Scene CI95 | Expressions/Scenes | Harm |
|---|---|---|---|---|---|
{intervals}
Official unique/multiple metadata unknown. Trained serial costs on frozen
TRAIN8 plus maxpair728/6780, two warmups/three readings; independent native
batch2 loss/backward/AdamW stress in memory only, no checkpoint overwritten:
```json
{json.dumps(cost['summary'], indent=2)}
```
Parameters {json.dumps(cost['parameters'])}; native128/oneargmax.
Serial descriptive p95 and one trained checkpoint do not establish Pareto.

## 5. Conclusion
Target unmet; {len(harm)} predeclared subgroup harm triggers. All relevant
intervals crossing zero remain inconclusive, not equivalence or no harm.
Retain W2; no Y37 promotion, retuning, renamed rerun or whole-family ban.

## 6. Crash Info
No training crash repair or restart. Bounded original final30/30 log saves
lastcheckpoint with no bounded traceback/OOM; not whole-log crash absence.
First evaluator invocation failed child train import before evaluation.
Its raw log is retained; retry inherited paths, same scientific evaluator AST
and checkpoints. Only evaluator's wrong GPU1 lease was delegated to the
authorized outer GPU0 lease; no sampler, metric or population change.

## Evidence
{json.dumps([evidence(p) for p in refs], indent=2)}
""")

names = ("boundary_cpu_nonempty", "native_controls", "physical_native",
         "initial_loss", "resources", "source_bundle")
paths, results = {}, {}
for name in names:
    paths[name], results[name] = load("y41_" + name)
cpu, native, physical, resource = (
    results["boundary_cpu_nonempty"], results["native_controls"],
    results["physical_native"], results["resources"])
assert cpu["cpu_cases"] == 32 and cpu["exact_binary_w2_max"] == 0
assert native["added_trainable_tensors"] == 4 and native["minimum_added_gradient"] > 0
assert native["cached_repeat_max"] == native["nonindicator_loss_delta_max"] == 0
assert min(native["wrong_masks_min"], native["wrong_pairs_min"]) > 0
assert physical["native_optimizer_steps"] == 2
assert all(physical["resume_before_continuation"][k]["equal"]
           for k in ("model", "optimizer", "scheduler"))
assert all(r["disabled_parent_mask_max"] == r["disabled_parent_indicator_max"] ==
           r["disabled_parent_sample_mismatches"] == 0
           for r in physical["cached_inference_rows"])
assert resource["parameters_added"] == 67074
assert resource["parameters_added"] < .01 * resource["summary"]["parent"]["parameters"]
assert resource["summary"]["candidate"]["stress_train_reserved"] < 24
assert not results["initial_loss"]["source_weights_or_production_checkpoint_changed"]
goal_bytes = (ROOT / "pipeline/goal.json").read_bytes()
assert hashlib.sha256(goal_bytes).hexdigest() == "53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966"
goal = json.loads(goal_bytes)
motivation_path = "pipeline/reviews/motivation_review_Y41_5e172bd4.md"
design_path = "pipeline/designs/exp_design_Y41_5e172bd4.md"
motivation = save(motivation_path, f"""# Motivation Review - AdaRES_project
Autonomous evidence review, not human approval or backdated preregistration.

## 1. Goal
Complete unchanged goal:
```json
{json.dumps(goal, indent=2)}
```
Target.515; automatic rounded championY36.5110/gap.0040; scientific W2 retained.

## 2. Story Status
| Hypothesis | State | Evidence | Gap |
|---|---|---|---|
| Y37 global distance confidence | {status} | full.505061022921; paired intervals cross0 | target unmet |
| Y40 missing direct word/support path | rejected realization | actualY35 CPU32 coverage | not a ban on language |
| Y41 boundary probability mass | native validation | equal-message CPU and TRAIN8/init4 | trained utility unknown |
| Y39 within-SP feature variance | training preserved | original102267/GPU1 | efficacy unknown |

## 3. Experiment Assessment
| Experiment | mIoU | Relation to motivation | Contribution |
|---|---|---|---|
| W2 | .510107995189 | metric-aligned query competition | current frozen comparator |
| B1 | .502119335285 | faithful standard indicator | full baseline comparator |
| Y37 | .505061022921 | global pair geometry confidence | no confirmed benefit |
| Y36 | prior.510716624292 | normalized local edge features | W2 superiority inconclusive |

## 4. Gap Analysis
Y36's nonempty independently normalized boundary/interior visual pools can
discard probability-weight totals. CPU keeps their visual messages equal
while new boundary descriptors separate; native controls show the new path
is reached. Existing learned features may indirectly encode these quantities.
No population-wide confidence bottleneck, upstream blindness or necessity
has been proven.

## 5. Design Rationale
Test explicit probability flow across nearest spatial edges plus normalized
Dirichlet energy at the final head, not a neighbor-radius/value search.
Native losses initially rise {json.dumps(native['loss_deltas'])}.
Fixed two new-path-only AdamW steps make mean losses decrease in both correct
and wrong-input conditions; two of eight correct TRAIN scenes rise.
These adverse/negative observations are retained. Learnability is not
necessity or efficacy; diagnostic weights are not deployed. Both failed CPU
controls are retained: finite roundoff and hard alternating empty interior.
The nonempty arithmetic control changes no gate tolerance. No literature
search or novelty claim; peer-review contribution remains unknown.

## 6. Next Batch
See {design_path}; only one fixed30 single-GPU candidate after six real gates
and an exact tested seven-file commit. Preserve Y39/GPU1.
""")
design = save(design_path, f"""# Experiment Design - AdaRES_project
Status VALIDATED_FOR_ADMISSION, not efficacy, human approval or launch.
Upstream motivation {motivation_path}; prospective prereg and repairs retained.

## 1. Motivation
{goal['description']}
Test whether final confidence benefits from boundary probability mass omitted
by normalized local messages. Begin active1/target2/budget1 after Y37 terminal.

## 2. Current Best Results
| ID | Config | State | mIoU | Acc50 | spmIoU | Epochs |
|---|---|---|---|---|---|---|
| Y37 | adares_y37_pair_distribution_res | done_complete | .505061022921 | .561001262095 | .468443808443 | 30 |
| W2 | adares_w2_pointmass_competition_res | comparator | .510107995189 | .562999579302 | .473719922153 | 30 |
| B1 | adares_b1_standard_res | baseline | .502119335285 | .552797644089 | .466681173553 | 30 |
Full FIRST captures and frozen terminal review are primary, not selected repeats.

## 3. Analysis & Direction
W2 remains comparator, not a causal mechanism proof. Y37 versus W2 and B1
intervals include zero, no promotion. Y40 proposed language realization is
already present in Y35. Y41 tests a distinct probability-edge information
path; absence of proven benefit remains explicit. Phase2 is not approved.

## 4. Experiment Plan
| ID | Config | Hypothesis | Expected result | Priority | Depends |
|---|---|---|---|---|---|
| AdaRES-Y41 | configs_song/model_paper5/adares_y41_boundary_probability_res.yaml | final boundary mass informs confidence | trained result may confirm or falsify; no promised gain | P0 | six gates |
Fixed native256-dimensional query plus3 scalars;259->256->2, four tensors.
Average exact nearest-distance ties; all valid SPs; no radius/feature subset,
new loss, threshold, temperature, margin, epoch or validation search.
Released SPFormer, fixed30/seed1999/original128/oneargmax/native optimizer.
Posttraining candidate/W2/B1 EACH9508/141 twice, FIRSTprimary; scene bootstrap
10000seed1999; TRAINcuts1487/3325; subgroupCIupper<0 versus either is harm.
Quantify empirical repeat noise and trained paired costs before promotion.
Official unique/multiple labels unknown if unavailable; target.515 unchanged.

## 5. GPU Allocation Plan
| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server2 | 1 | preserve Y39 | adares_y39_point_distribution_res | existing original job |
| server2 | 0 | proposed Y41 | adares_y41_boundary_probability_res | only fresh idle verified slot |
No server6 or foreign project operations. Canonical FIFO dispatcher limited to0.

## 6. Implementation Checklist
- [x] Physical independent six-file package/config, no cross-variant imports.
- [x] CPU32, nativeTRAIN8/init4 wrong controls, finite four-tensor gradients.
- [x] Actual tools.train two optimizer steps and model/AdamW/PolyLR restoration.
- [x] Disabled physical W2 masks/indicators/sampling identical on four scans.
- [x] Fixed initialized resource screen; source/checkpoint preservation.
- [ ] Commit exactly seven tested files, verify Git/local/remote bytes.
- [ ] Canonical six-gate receipt and scoped GPU0 dispatch; no blind restart.
- [ ] Independent increasing native training steps before training status.

## 7. Risk Assessment
Spatial graph/scalar information may correlate with existing learned features
or overfit. Initial mean loss rises and wrong controls also learn. Direct
edge squared differences avoid cancellation but nearest ties can be dense;
actual max-scene stress is reported, not a universal bound.
Restored state exact; next continuation model/optimizer differs numerically.
Resource summary {json.dumps(resource['summary'])}. Added67074<1% and peak<24GiB
are initialized feasibility only, not trained Pareto superiority.
All goal constraints/bans above are preserved. No GT at inference, schema,
data/schedule/loss changes, background scheduler, multiGPU or other project.

## 8. Smell Check
Not signal1:67074parameters and a new nearest-edge probability flow graph.
Potential signal2, 'add a Linear', is rebutted by the new nontrivial graph
descriptor, not an affine modification of existing identical inputs.
Not signal3: the information path changes; not just scaling or normalization.
No signals4/5/6: no numeric/configuration scan, banned-route rename or value
search. Width256 is inherited; exactly3 statistics are mechanistically defined.
Causal claim is falsifiable usefulness of explicit edge probability mass,
not 'a better coefficient'. Novelty/contribution and trained benefit unknown.
""")
review_path = "pipeline/analyses/Y41_admission_review_5e172bd4.md"
review = save(review_path, f"""# Y41 admission-support review
Observed {datetime.now().astimezone().isoformat()}. Six operational gates pass
the actual evidence below; no efficacy or human approval is asserted.
CPU failures, finite roundoff, empty-pool existing-path response, initial
adverse loss, wrong-control learning and continuation deviations are retained.
{json.dumps([evidence(p) for p in paths.values()], indent=2)}
Motivation {json.dumps(motivation)}; design {json.dumps(design)}.
""")
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    y37 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y37")
    assert y37["stage"] == "training"
    y37.update(stage="rejected" if harm else "complete", scientific_status=status,
               next_action="Retain W2; no renamed Y37 rerun or validation retuning. "
                           "New independent evidence required before reconsideration.")
    y37["evidence"] += [y37_review] + [evidence(p) for p in refs]
    y41 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y41")
    assert y41["stage"] == "diagnosis"
    y41.update(stage="validation", scientific_status="validated_not_efficacy",
               next_action="Commit exactly tested seven files, verify byte identity, "
                           "then canonical receipt/dispatch and independent increasing "
                           "steps on GPU0 while preserving Y39/GPU1.")
    repair_paths = [
        "pipeline/analyses/Y41_resource_freeze_5e172bd4.md",
        "pipeline/analyses/Y41_cpu_control_repair_5e172bd4.md",
        "pipeline/analyses/Y41_nonempty_control_repair_5e172bd4.md",
        "pipeline/analyses/y41_boundary_cpu_" + TOKEN + ".log",
        "pipeline/analyses/y41_boundary_cpu_audit_" + TOKEN + ".log",
    ]
    y41["evidence"] += [evidence(p) for p in paths.values()] + [
        motivation, design, review] + [evidence(p) for p in repair_paths]
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
finally:
    lock.close()
print(json.dumps({"y37_status": status, "y37_review": y37_review,
                  "y41_review": review, "y41_stage": "validation"}, indent=2))
