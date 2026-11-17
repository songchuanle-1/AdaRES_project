"""Persist the evidence review and same-ID progress without changing gates."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

base = "pipeline/analyses/"
names = ("pooling_cpu_v2", "train_ceiling", "readout_cpu", "extract_cpu", "native_controls",
         "physical_native", "initial_loss", "resources", "source_bundle", "local_bundle")
paths = {name: base + "y39_" + name + "_" + args.token + ".json" for name in names}
results = {name: json.loads((ROOT / path).read_text(encoding="utf-8")) for name, path in paths.items()}
assert all(result["complete"] for result in results.values())
goal_bytes = (ROOT / "pipeline/goal.json").read_bytes()
goal_sha = hashlib.sha256(goal_bytes).hexdigest()
assert goal_sha == "53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966"
goal = json.loads(goal_bytes)
native = results["native_controls"]
resource = results["resources"]
investigation = results["initial_loss"]
assert native["added_trainable_tensors"] == 8 and native["minimum_added_gradient"] > 0
assert native["cached_repeat_max"] == native["nonindicator_loss_delta_max"] == 0
assert resource["parameters_added"] == 271618
assert resource["summary"]["candidate"]["stress_train_reserved"] < 24
assert resource["parameters_added"] < .01 * resource["summary"]["parent"]["parameters"]
assert not investigation["source_weights_or_production_checkpoint_changed"]
physical = results["physical_native"]
assert physical["native_optimizer_steps"] == 2
assert all(physical["resume_before_continuation"][key]["equal"] for key in ("model", "optimizer", "scheduler"))

motivation_path = "pipeline/reviews/motivation_review_Y39_48678b81.md"
design_path = "pipeline/designs/exp_design_Y39_48678b81.md"
review_path = base + "Y39_admission_review_48678b81.md"
review = """# Motivation Review - AdaRES_project
This review consolidates already frozen diagnosis/realization and completed
controls. It is not retroactively described as a preregistration. Autonomous
scientific review, not a human approval.

## 1. Goal
Unchanged complete goal.json, SHA""" + goal_sha + """:
```json
""" + json.dumps(goal, indent=2, ensure_ascii=True) + """
```
Target .515. Automatic snapshot ranks rounded Y36 .5110, gap .0040.
Frozen scientific review retains W2: Y36 superiority interval crosses zero.
Phase1 unmet; Phase2 unapproved.

## 2. Story Status
| Hypothesis | State | Evidence | Gap |
|---|---|---|---|
| W2 point-mass competition | retained comparator | full .510009849923 | target unmet, causality not proven |
| Y36 local adjacency | complete_inconclusive | full .510716624292, paired W2 CI crosses0 | no superiority/promotion |
| Y37 global pair distribution | original training preserved | snapshot10/30, rounded .4926 | trained outcome unknown |
| Y38 missing mask history | tested rejection of realization | CPU32 actual Y30 minimum response .005445152521 | proposed missing path already implemented |
| Y39 within-SP feature heterogeneity | native path separated | frozen equal-mean CPU and TRAIN8/init4 | trained utility unknown |

## 3. Experiment Assessment
| Experiment | mIoU | Relation to motivation | Contribution |
|---|---|---|---|
| B1 | .502112417447 | faithful baseline confidence | fixed comparison baseline |
| W2 | .510009849923 | explicit competition evidence | retained trained comparator |
| Y36 | .510716624292 | local adjacency confidence | W2 difference+.000706774369, CI[-.007218154973,.008880424700]; inconclusive |
| Y35 | prior .504801213799 | language evidence confidence | inconclusive prior frozen review, no retest here |
| Y30 | not reloaded | existing explicit mask history/GRU | old medium-harm record preserved; direct new CPU coverage rejects Y38 |

## 4. Gap Analysis
Native W2 averages all32 point features per superpoint. A distribution with
identical means/counts/centroids may have different within-SP variance, while
the final head receives only pooled features and predicted masks/mass.
This is a direct-path information gap, not a claim that upstream learned
means or attention are globally blind to heterogeneity.
TRAIN64/256 exact whole-superpoint mask representability mean .999771170370,
minimum .989761092150, median1; only29 expressions with nonzero gap.
This NEGATIVE diagnosis does not support quantization as the main task
bottleneck. Y39 instead tests learned confidence from point-feature
heterogeneity; GT is never an inference input.

## 5. Design Rationale
All32 centered point-feature variance is computed before native pooling,
then final soft-mask foreground/background point-mass-weighted encoded
distributions inform the existing final indicator. This adds a nonlinear
distribution path without searching thresholds/features/loss settings.
The actual frozen class reacts to wrong SP variance association and wrong
masks, is repeatable and padding-safe, and receives native gradients.
Initial trained-W2 mean loss rises; every adverse/control loss is retained.
Fixed two-step new-path-only AdamW investigates correctability, not necessity
or validation benefit. All wrong-input controls can learn too.
No external literature/novelty search: novelty unknown. No trained efficacy,
no harm, semantic necessity or Pareto claim.

## 6. Next Batch
See """ + design_path + """.
One admission at most, only after actual six-gate review and tested code commit.
"""
description = goal.get("description", "See unchanged complete goal in motivation review.")
design = """# Experiment Design - AdaRES_project
Status: VALIDATION; no admission or launch is asserted here.
Tick48678b818b134bea9d9c3c3e1f3d3305, original deadline20:48:21+08 unchanged.

## 1. Motivation
""" + description + """
Upstream: """ + motivation_path + """.
Stable diagnosis/realization frozen before controls; this document consolidates
those records, not a newly backdated freeze. Capacity2/active1/budget1 after
canonical Y36 terminal closure. Preserve original Y37/GPU0.

## 2. Current Best Results
| Experiment ID | Config | State | Best mIoU | Acc@50 | spmIOU | Epochs |
|---|---|---|---|---|---|---|
| W2 | adares_w2_pointmass_competition_res.yaml | retained frozen comparator | .510009849923 | not loaded | not loaded | 30 |
| B1 | not loaded here | frozen baseline | .502112417447 | not loaded | not loaded | 30 |
| Y36 | adares_y36_local_adjacency_res.yaml | done_complete/inconclusive | .510716624292 | not loaded | not loaded | 30 |
| Y37 | adares_y37_pair_distribution_res.yaml | preserved training | snapshot rounded .4926 | unknown | unknown | monitor10/30 |
These full results are reused from prior frozen reviews, not rerun today.

## 3. Analysis And Direction
### 3.1 Observed Positive Results
W2 improves on B1 in these fixed trained references. Y36's B1 interval is
positive but its W2 interval crosses0; not a necessity/superiority result.
### 3.2 Negative Or Unresolved Results
Actual Y30 history path already covers Y38. Native partition ceiling nearly1
disconfirms quantization as a major bottleneck. Sparse quantization is not an
arbitrary pretraining magnitude veto, but its limitation remains explicit.
### 3.3 Not Yet Verified
Whether within-SP feature heterogeneity generalizes as confidence evidence;
learned upstream means may already encode it. No Phase2 ablations authorized.
### 3.4 Selected Direction
Y39's explicit native point-feature distribution is separated on equal-mean
fixtures and matched native wrong-input controls; only falsifiable training
can decide benefit. It is not a renamed global geometry/history route.

## 4. Experiment Plan
| # | ID | Config Path | Purpose / Hypothesis | Expected Result | Priority | Depends On |
|---|---|---|---|---|---|---|
| 1 | AdaRES-Y39 | configs_song/model_paper5/adares_y39_point_distribution_res.yaml | final point-distribution confidence | reproducible mechanism response, then frozen trained quality/harm/cost; no promised gain | P0 | six gates |
Population centered variance across ALL32 raw backbone point features in each
native superpoint; log1p then shared32->256->256/GELU encoder. Detached final
soft masks and native point masses give FG/BG means. Final query, encoded
FG-BG, encoded(FG+BG)/2 ->768->256->2 indicator residual. Eight tensors,
271618 added parameters. No auxiliary head, mask output, loss weight,
query count, point input subset, inference argmax or native schedule changes.
Production from released SPFormer, seed1999, fixed30. Trained W2 weights
are diagnosis/comparison only, never production initialization.
Post-training frozen full validation9508/141, Y39/W2/B1 each two passes,
FIRST primary. Whole-scene10000 bootstrapseed1999 against both comparators.
TRAIN point-size cuts1487/3325; subgroup material-harm rule CIupper<0 against
either comparator. CIcross0 inconclusive, not equivalence/noharm. Measure
repeat noise and trained paired costs, no invented effect/noise threshold.
Target.515 and frozen harm/lightweight Pareto review required for promotion.

## 5. GPU Allocation Plan
| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server-2 | 0 | preserve original Y37 | adares_y37_pair_distribution_res.yaml | no restart, migration or source overwrite |
| server-2 | 1 | exclusive Y39 diagnostics; possible gated FIFO admission | adares_y39_point_distribution_res.yaml | fresh live idle check, no future-idle assumption |
Single GPU, one job per GPU; no server6 or other-project/scheduler operations.

## 6. Implementation Checklist
Own six-file physical package and parent-exact YAML created. Scoped additive
upload with all Y37 source hashes unchanged. CPU32, nativeTRAIN8/init4,
native tools.train2steps/all8updated/checkpoint restoration/physicalW2
disabled identity and fixed resource measurements completed.
Adverse native-loss investigation retained. Commit ONLY own sevenfiles,
git/local/remote exact-byte alignment and admission/dispatch/increasing
native launch steps still pending here. No unrelated dirty-file changes.

## 7. Risk Assessment
All32 variance may be an indirect appearance proxy, not uniquely sufficient
or necessary confidence evidence. Near-perfect partition ceilings are a
negative result and cannot establish expected trained gain.
Random new-head calibration initially worsens trained W2 loss. Correct and
wrong controls can all learn; mean correction is not causal generalization.
Point variance adds native backbone gradients although masks are detached
only in the new readout. Original losses and paths remain. CUDA scatter
accumulation and sparse continuation have numerical noise, not bitwise whole
trajectory reproducibility. Actual time/memory regressions retained.
Feasibility addedparams<1%/peakreserved<24GiB; trained Pareto still unknown.
No tuning, new data, GT inference, threshold/temperature/neighborhood search.

## 8. Smell Check
| Signal | Judgment |
|---|---|
| <5K/no new graph | no:271618 parameters plus raw-point centered aggregation |
| norm/linear/gate only | projections present but new per-SP distribution path before pooling |
| numeric-only variant | no: physical native input path and masked distribution aggregation |
| sweep | none; inherit native32/256, no dimension/feature/loss search |
| renamed banned route | not threshold/margin/radius/history/global covariance; explicit within-SP feature spread |
| equivalence to old head | W2 final state held equal in CPU fixtures while actual new readout changes |
These checks support executability only, not novelty or efficacy.
"""
assessment = """# Y39 Admission Evidence Review
Autonomous scientific review before any admission, not human authorization
or a trained result. No missing test is made PASS by this text.

## Actual Measurements
""" + json.dumps(dict(
    controls={key: native[key] for key in ("minimum_added_gradient", "cached_repeat_max",
                                          "nonindicator_loss_delta_max", "wrong_masks_min",
                                          "wrong_variance_min", "permutation_max", "loss_deltas")},
    physical={key: physical[key] for key in ("native_optimizer_steps", "continuation")},
    resource=resource["summary"],
    adverse_investigation={phase: {mode: outcome["mean_change"] for mode, outcome in value.items()}
                           for phase, value in investigation["summary"].items()},
), indent=2) + """

## Judgment
Diagnosis: explicit all32 point-distribution information lost by direct mean
readout, native incidence nonzero, matched no-GT fixtures. Partition ceiling
nearly1 does not establish a major quantization deficit.
Controls: actual native associations change final confidence while disabled
physical W2 masks/indicators/sampling are identical. Repeat and nonindicator
loss deltas0, padding and bounded permutation checks pass. Wrong controls'
losses are not evidence of causal benefit.
Native backward: native tools.train2steps, all8finite/nonzero/updated; exact
model/AdamW/PolyLR restoration before continuation, numerical next-step
differences retained.
Resources: measured parameter/memory feasibility with fixed paired protocol.
Latency regressions are descriptive; initialized feasibility is not trained
quality/cost Pareto. Independence: own physical sixfiles/config, no cross-
variant imports, original Y37 unchanged. Constraints: unchanged goal/fixed
native training protocol. Commit/exact bytes and gated dispatch still pending.
Native initial W2 loss increase was investigated with only the new path
temporarily optimized using fixed original AdamW and two steps. All correct/
wrong outcomes retained; diagnostic weights never deployed.
Correct-input trained mean changed by -.001247614622, but5of8 individual
scene losses increased; correctability is not a universal correction.
The resource protocol inherited a stale word "trajectory" in its limitations;
the actual resolved Y39 package is point-feature variance, not mask history.
Original result is retained, not rewritten to hide that wording defect.
Scientific utility, necessity, novelty, harm and trained Pareto UNKNOWN.
Only admission for falsifiable training can follow; not scientific promotion.
"""
extra = [
    base + "Y39_diagnosis_prereg_48678b81.md",
    base + "Y39_realization_freeze_48678b81.md",
    base + "Y39_resource_freeze_48678b81.md",
    "scripts/probe_y39_pooling_cpu.py", "scripts/probe_y39_pooling_cpu_v2.py",
    "scripts/probe_y39_readout_cpu.py", "scripts/probe_y39_extract_cpu.py", "scripts/probe_y39_train_ceiling.py",
    "scripts/probe_y39_native_controls.py", "scripts/probe_y39_physical_native.py",
    "scripts/probe_y39_initial_loss_investigation.py", "scripts/probe_y39_resources.py",
]
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    for relative, text in ((motivation_path, review), (design_path, design), (review_path, assessment)):
        path = ROOT / relative
        assert not path.exists()
        path.write_text(text, encoding="utf-8", newline="\n")
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidate = next(item for item in work["candidates"] if item["id"] == "AdaRES-Y39")
    assert candidate["stage"] == "diagnosis"
    refs = list(paths.values()) + extra + [motivation_path, design_path, review_path]
    existing = {item["path"] for item in candidate["evidence"]}
    for relative in refs:
        if relative not in existing:
            candidate["evidence"].append(dict(path=relative, sha256=hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()))
    failed_log = base + "y39_pooling_cpu_" + args.token + ".log"
    assert (ROOT / failed_log).exists()
    candidate["evidence"].append(dict(path=failed_log, sha256=hashlib.sha256((ROOT / failed_log).read_bytes()).hexdigest()))
    candidate.update(stage="validation",
                     next_action="Verify exact candidate git/local/remote byte identity after candidate-only commit; "
                                 "then actual six-gate admission/canonical GPU1 dispatcher and increasing native launch steps.",
                     scientific_status="validation_not_trained_efficacy",
                     gate_review=review_path,
                     limitations="TRAIN partition mean ceiling .999771; initial trained-W2 adverse loss preserved; "
                                 "wrong controls also learn; novelty, trained benefit, harm and Pareto unknown.")
    work_path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(id=candidate["id"], stage=candidate["stage"],
                          evidence_count=len(candidate["evidence"]),
                          next_action=candidate["next_action"]), indent=2))
finally:
    lock.close()
