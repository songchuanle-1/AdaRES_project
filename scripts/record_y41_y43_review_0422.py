"""Close Y41 from frozen evidence and review actual Y43 feasibility, not efficacy."""

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


def evidence(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


def load(name):
    path = "pipeline/analyses/" + name + "_" + args.token + ".json"
    result = json.loads((ROOT / path).read_text())
    assert result["complete"]
    return path, result


names = ("y41_terminal_0422", "y41_frozen_full_0422", "y41_clustered_comparison",
         "y41_object_groups", "y41_trained_cost_0422_v2")
terminal_results = {name: load(name) for name in names}
terminal, full, analysis, objects, cost = [terminal_results[n][1] for n in names]
assert not analysis["target_reached"]
assert all(not group["frozen_material_harm_ci_upper_below_zero"]
           for ref in ("versus_w2", "versus_b1") for group in analysis[ref].values())
best = next(c for c in terminal["checkpoints"] if c["path"].endswith("/best.pth"))
y41_review = "pipeline/analyses/AdaRES-Y41_frozen_review_0422d0d8.md"
y43_review = "pipeline/analyses/Y43_six_gate_review_0422d0d8.md"
assert not (ROOT / y41_review).exists() and not (ROOT / y43_review).exists()
candidate = analysis["summary"]["y41"]
paired = analysis["versus_w2"]["all"]
y41_text = f"""# Experiment Result Analysis: AdaRES-Y41
Observed {datetime.now().astimezone().isoformat()}.

## 1. Basic Info
Canonical done_complete. Original PID116906 absent; GPU0. Config
adares_y41_boundary_probability_res.yaml. Start/exit/duration unknown.
Original last30/best{best['epoch']}; bestSHA{best['sha256']}.
Checkpoints/model/optimizer/scheduler read-only. Evaluation9508/141 twice
EACH Y41/W2/B1; FIRST primary, native128 queries/one argmax.

## 2. Best Metrics
| Metric | FIRST Frozen Full | Checkpoint Epoch |
|---|---|---|
| mIoU | {candidate['miou']} | {best['epoch']} |
| Acc25 | {candidate['acc25']} | {best['epoch']} |
| Acc50 | {candidate['acc50']} | {best['epoch']} |
| spmIoU | {candidate['spmiou']} | {best['epoch']} |
Original best metadata{best['best_miou']} is not a replacement for frozen evaluation.

## 3. Convergence Summary
Fixed30 completed; best first saved at21. Final rounded log mIoU.502.
This is not evidence of an optimal epoch or convergence; no retuning/resume.

## 4. Comparison
Y41-W2{paired['paired_miou_difference']}/scene95CI{paired['scene_clustered_ci95']}.
Y41-B1{analysis['versus_b1']['all']['paired_miou_difference']}/
CI{analysis['versus_b1']['all']['scene_clustered_ci95']}.
Whole-scene10000 bootstrap/seed1999/TRAINcuts1487/3325. No observed all/size
upper<0 harm trigger; crossing0 is inconclusive, not equivalence/no-harm.
Medium versus W2 point estimate{analysis['versus_w2']['medium']['paired_miou_difference']}
is adverse. Small versus B1 interval is positive, not overall superiority.
Single-object9508/141; multi-object0/0 insufficient. Official unique/multiple
unknown; object count is not a substitute.
Per-model repeat maxima{json.dumps({k:v['repeat_max_iou_difference'] for k,v in analysis['cost_first_chunk_only'].items()})};
aggregate repeat means{json.dumps(analysis['repeated_miou'])}.
Trained paired cost{json.dumps(cost['summary'])}; parameters{json.dumps(cost['parameters'])}.
Frozen TRAIN8 plus pressurepair728/6780,2warm/3read; temporary native updates
memory-only. Candidate-then-parent order/descriptive p95/single checkpoint
limit cost interpretation. No trained Pareto superiority.

## 5. Conclusion
SAME Y41 complete_inconclusive. Target.515 unmet; retain W2. No promotion,
retune/resume or renamed equivalent. Native confidence boundary route remains
unproven. All original evidence and failed cost-adapter log preserved.

## 6. Crash Info
No canonical training crash. Cost-adapter first attempt failed before model
execution due to protected-job bytes; v2 corrected Y42 GPU1 binding only.
Bounded tail no traceback/OOM is not whole-log proof. Original Y42 PID91215/
GPU1/source retained. Hashed evidence:
{json.dumps([evidence(v[0]) for v in terminal_results.values()] + [evidence('pipeline/analyses/y41_cost_binding_repair_' + args.token + '.json')])}
"""

y43_names = ("y43_reflection_cpu_0422", "y43_extract_readout_cpu_0422",
             "y43_parent_semantics_0422", "y43_source_bundle",
             "y43_native_controls_0422_v2", "y43_physical_native_0422",
             "y43_initial_loss_0422", "y43_resources_0422")
y43_results = {name: load(name) for name in y43_names}
cpu = y43_results["y43_extract_readout_cpu_0422"][1]
native = y43_results["y43_native_controls_0422_v2"][1]
physical = y43_results["y43_physical_native_0422"][1]
resources = y43_results["y43_resources_0422"][1]
initial = y43_results["y43_initial_loss_0422"][1]
assert cpu["cpu_cases"] == 32 and cpu["all_fixture_features_nonnegative"]
assert cpu["minimum_reflection_delta"] > cpu["permutation_floor"]
assert cpu["all_zero_moment_gradients_finite"]
assert native["added_trainable_tensors"] == 8 and native["minimum_added_gradient"] > 0
assert native["cached_repeat_max"] == native["nonindicator_loss_delta_max"] == 0
assert min(native["wrong_masks_min"], native["wrong_third_min"], native["reflected_third_min"]) > 0
assert all(row["padding_poison"] == 0 and row["detached_mask_gradient"]
           for row in native["cached_controls"])
assert y43_results["y43_parent_semantics_0422"][1]["whole_modules_parent_identity"]
assert physical["native_optimizer_steps"] == 2
assert all(physical["resume_before_continuation"][k]["equal"] for k in ("model", "optimizer", "scheduler"))
assert all(r["disabled_parent_mask_max"] == r["disabled_parent_indicator_max"]
           == r["disabled_parent_sample_mismatches"] == 0 for r in physical["cached_inference_rows"])
assert resources["parameters_added"] == 271618 < .01 * resources["summary"]["parent"]["parameters"]
assert resources["summary"]["candidate"]["stress_train_reserved"] < 24
assert not initial["source_weights_or_production_checkpoint_changed"]
y43_text = f"""# Y43 Six-Gate Feasibility Review
Observed {datetime.now().astimezone().isoformat()}; not efficacy or human approval.

## 1. Diagnosis
Actual32-seed existing mean/variance/full-covariance reflection readouts delta0.
New actual independent branch nonnegative32-seed witness minimum reflection
response{cpu['minimum_reflection_delta']}, controlfloor{cpu['permutation_floor']}.
Native prediction-blind TRAIN8/init4 nonzero third statistics recorded.
Narrow explicit-statistic gap only; upstream encoding/task utility unknown.
Initial witness permitted negative reflected features; positive-codomain
confirmation retained separately, not an overwritten old result.

## 2. Controls
Actual wrongmask/rolledthird/reflection responses nonzero; repeat, padding
poison and nonindicator loss changes0, detached masks, permutations bounded.
Raw reflected third-summary intervention is a controlled feature-statistic
counterfactual, not a claim that native ReLU activations reflect identically.
Native initial loss deltas{json.dumps(native['loss_deltas'])} retained.
Fixed2 new-only AdamW steps{json.dumps({phase:{mode:value['mean_change'] for mode,value in outcomes.items()} for phase,outcomes in initial['summary'].items()})}.
Wrong controls can learn; no necessity/generalization/benefit conclusion.
All scene-specific adverse changes retained. Diagnostic weights not deployed.

## 3. Native Backward
Actual tools.train2 steps; all8 new tensors update with finite nonzero native
gradients; exact model/AdamW/PolyLR restoration before continuation.
Continuation{json.dumps(physical['continuation'])}; no bitwise trajectory claim.
Four cached scenes: disabled-path physical W2 mask/indicator/selection equal0.
Native128/one argmax remains unchanged. Not whole-loader determinism.

## 4. Resources
New271618 params; native pressure reserved<24GiB and no nonfinite/OOM passed.
Paired frozen TRAIN8/pressurepair728,6780 two warmups/three readings:
{json.dumps(resources['summary'])}.
Report observed time regression, not trained Pareto superiority. Prediction
uses W2 weights plus freshly initialized branch, not trained Y43.

## 5. Independence
Seven local/remote source hashes equal; unique physical package/YAML, no
cross-variant imports. Same-Python whole dec/ipdn AST equals W2 after stripping
exact new path, configdiffs[]. Shared initial tensors match. Native disabled
parent equality also tested. Original Y42 PID91215/GPU1/source remains intact.
Only new seven files LF-formatted before tests. Two pre-execution adapter
path corrections, first failed Y41 cost binding and failed Y43 native result
keyword adapter retained; v2 changes only recorded field names. Model/gates/
tolerances not changed after native observations.
Commit/raw git-blob binding and canonical dispatch still required.

## 6. Constraints And Disposition
Six actual executability/falsifiability gates support Phase1 fixed30 training.
No trained efficacy, necessity, novelty, equivalence, no-harm or human approval.
W2 retained/.515 unmet/Phase2 unapproved. Released backbone/nativeoptimizer/
loss/scheduler/seed1999/native128/argmax1 unchanged. No subset/neighbors/
threshold/temperature/weights/epochs/validation sweep.
Terminal evaluation remains full9508/141 twice EACH/FIRST/scenebootstrap/
size/object/harmvsEither/noise/trainedpairedcost as preregistered.
Hashed evidence:
{json.dumps([evidence(v[0]) for v in y43_results.values()])}
"""

lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    (ROOT / y41_review).write_text(y41_text, encoding="utf-8")
    (ROOT / y43_review).write_text(y43_text, encoding="utf-8")
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text())
    y41 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y41")
    y43 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y43")
    assert y41["stage"] == "training" and y43["stage"] == "diagnosis"
    y41.update(stage="complete", scientific_status="complete_inconclusive",
               result=f"Frozen FIRST{candidate['miou']}, paired W2 CI crosses0; target unmet.",
               next_action="Retain W2 and immutable evidence; no restart/retune/promotion.")
    y43.update(stage="validation", scientific_status="six_gate_feasibility_not_efficacy",
               next_action="Commit exactly seven tested files, bind receipt and canonical GPU0 dispatch; verify increasing original native training steps.")
    refs41 = [v[0] for v in terminal_results.values()] + [y41_review,
        "pipeline/analyses/y41_cost_binding_repair_" + args.token + ".json"]
    refs43 = [v[0] for v in y43_results.values()] + [y43_review,
        "pipeline/designs/exp_design_Y43_0422d0d8.md",
        "pipeline/reviews/motivation_review_Y43_0422d0d8.md",
        "pipeline/analyses/y43_lf_and_preservation_" + args.token + ".json",
        "pipeline/analyses/y43_native_field_repair_" + args.token + ".json"]
    for candidate_row, refs in ((y41, refs41), (y43, refs43)):
        for ref in refs:
            entry = evidence(ref)
            assert not any(e["path"] == ref for e in candidate_row["evidence"])
            candidate_row["evidence"].append(entry)
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"y41": y41["scientific_status"], "y43": y43["scientific_status"],
                      "reviews": [evidence(y41_review), evidence(y43_review)]}))
finally:
    lock.close()
