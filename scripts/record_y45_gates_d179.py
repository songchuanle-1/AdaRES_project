"""Review actual executability gates, retaining native objective conflicts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "392a5a9f5fe242d7a3109a50e6060fda"
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])


def load(stem, token=PRIOR):
    path = ROOT / ("pipeline/analyses/" + stem + "_" + token + ".json")
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["complete"]
    return path, value


def evidence(path):
    path = path if isinstance(path, Path) else ROOT / path
    return dict(path=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


native_path, native = load("y45_native_targets_392a")
objectives_path, objectives = load("y45_decomposition_controls_392a")
cpu_path, cpu = load("y45_package_cpu_392a_v2")
controls_path, controls = load("y45_native_path_392a_v2")
physical_path, physical = load("y45_physical_native_392a")
resources_path, resources = load("y45_resources_binding_392a")
learning_path, learning = load("y45_initial_learning_392a")
bundle_path, bundle = load("y45_source_bundle_392a")
gradient_path, gradient = load("y45_gradient_d179", args.token)
assert native["scenes"] == 8 and native["expressions"] == 32 and native["point_target_max_error"] == 0
assert objectives["cases"] == 32 and objectives["all_closed_actual_loss_gradient_gaps_zero"]
assert objectives["minimum_decomposition_gradient_gap"] > 0
assert cpu["whole_parent_AST_identity"] and cpu["config_differences"] == []
assert cpu["minimum_wrong_mask_response"] > cpu["observed_control_floor"]
assert controls["disabled_physical_W2_equal"] and controls["literal_target_max_error"] == 0
assert controls["new_tensors"] == 4 and controls["minimum_native_new_gradient"] > 0
assert physical["native_optimizer_steps"] == 2 and len(physical["native_gradient_records"]) == 2
assert all(len(r["norm_gradients"]) == 4 and all(v > 0 for v in r["norm_gradients"].values())
           for r in physical["native_gradient_records"])
assert len(physical["peer_parameter_changes"]) == 4
assert all(v > 0 for v in physical["peer_parameter_changes"].values())
assert all(physical["resume_before_continuation"][k]["equal"] for k in ("model", "optimizer", "scheduler"))
assert resources["id"] == "AdaRES-Y45" and resources["raw_result_id"] == "AdaRES-Y43"
assert resources["added_parameters"] == 197635
assert resources["summary"]["candidate"]["stress_train_reserved"] < 24
assert resources["added_parameters"] < .01 * resources["summary"]["parent"]["parameters"]
assert learning["trained_scenes"] == 8 and learning["released_init_scenes"] == 4
assert learning["fixed_new_only_steps_per_control"] == 2 and not learning["diagnostic_weights_deployed"]
assert gradient["trained_scenes"] == 8 and gradient["released_init_scenes"] == 4
assert all(r["controls"]["repeat"]["repeat_gradient_max"] == 0 for r in gradient["rows"])
assert bundle["physical_independence"] and bundle["remote_byte_identity"] and len(bundle["files"]) == 7
for row in bundle["files"]:
    assert evidence(row["path"]) == row
assert evidence("pipeline/goal.json")["sha256"] == "53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966"
gates = {
    "diagnosis": [native_path, objectives_path],
    "controls": [cpu_path, controls_path, learning_path, gradient_path],
    "native_backward": [controls_path, physical_path],
    "resources": [resources_path],
    "independence": [bundle_path, cpu_path],
    "constraints": [ROOT / "pipeline/goal.json", ROOT / "pipeline/designs/exp_design_Y45_392a.md"],
}
now = datetime.now().astimezone().isoformat()
review_path = ROOT / "pipeline/analyses/Y45_six_gate_review_d179.md"
assert not review_path.exists()
review = f"""# SAME Y45 Executability Gate Review
Observed locally {now}; owned tick {args.token}.

| Gate | Result | Actual evidence | Limits |
|---|---|---|---|
| diagnosis | PASS executable information gap | native TRAIN8/32 literal point targets; actual B2/G1/C1 equal-IoU objective controls | no proof of hidden-state blindness or trained utility |
| controls | PASS intended intervention | actual-class CPU32 and cached native controls; correct/wrong target gradients; disabled physical W2 equality | full-input exact repeat not established; wrong controls also learn |
| native_backward | PASS graph/update/restore | four finite nonzero new gradients, native tools.train two real optimizer updates and exact restored model/AdamW/PolyLR | next-step numerical differences retained, not bitwise whole-loader continuation |
| resources | PASS fixed feasibility budget | added197635<1% W2; native reserved13.541GiB<24GiB/noOOM/nonfinite | costs initialized; raw Y43 export label clarified additively; no trained Pareto claim |
| independence | PASS physical package | seven independent files/config; local/remote raw hashes and stripped parent AST | no novelty conclusion |
| constraints | PASS unchanged protocol | releasedSPFormer/seed1999/fixed30/native128/argmax1/server2GPU1 | no validation tuning; no diagnostic weights deployed |

## Native Adverse-Result Interpretation
The prior native legacy loss increase is NOT discarded or excused by a new
effect-size threshold. Separate gradients reproduce an opposing local
direction in7/8 W2-trained scenes, versus0/4 released-init scenes. Correct
targets oppose4 matched/166 unmatched query directions; cyclic wrong targets
oppose152 matched/35 unmatched. The observed tradeoff is therefore source-
and target-attributed, not an unexplained silent path or relabeled scalar target.
These numbers are descriptive on the preregistered cohort, not population tests.
Exact native analytic derivatives and categorical=binary-IoU+conditional-FP/FN
factorization agree within the prospective8*float32eps envelope; repeats of
cached inputs have equal gradients. FP/FN contributes distinct conditional
supervision. This does not establish necessity, calibration or expected mIoU.

The released-initialization direction is relevant to the prospective production
run, which starts from released SPFormer, not from W2 trained/diagnostic weights.
Task efficacy and harm must be decided by the unchanged frozen post-training
plan, not these local gradient signs, tiny losses or a minimum gain cutoff.
An effect-size veto here would reintroduce the obsolete pre-training threshold.
No weights, targets, numeric settings, inference selector or historical failures
were changed to obtain this executability assessment.

## Frozen Evaluation And Disposition
Candidate design pipeline/designs/exp_design_Y45_392a.md remains unchanged:
fixed30/seed1999/singleGPU; each Y45/W2/B1 full9508/141 twice, FIRST primary;
scene bootstrap10000seed1999/TRAIN cuts1487,3325; meaningful subgroup CI
upper<0 versus either comparator triggers original harm review. Report repeats,
official unique/multiple unknown, active128/singleargmax and trained paired costs.
Commit only the seven tested files and issue a hash-bound admission receipt
before canonical GPU1 dispatch. A process alone is not verified training.
This review approves a falsifiable Phase1 training test, NOT efficacy,
necessity, novelty, no-harm, Phase2 approval or an expected positive result.
All original v1 audit/full-input-repeat failures and raw resource label remain.
"""
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    review_path.write_text(review, encoding="utf-8")
    checks_path = ROOT / ("pipeline/analyses/y45_gate_checks_d179_" + args.token + ".json")
    assert not checks_path.exists()
    checks = dict(complete=True, id="AdaRES-Y45", observed_at_local=now,
                  gates={key: True for key in gates},
                  evidence={key: [evidence(p) for p in paths] for key, paths in gates.items()},
                  review=evidence(review_path), scientific_status="executable_for_falsifiable_training_not_efficacy",
                  limits="Local conflicts and source failures retained; no effect-size threshold, weight tuning or diagnostic weight deployment.")
    checks_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y45")
    assert candidate["stage"] == "validation"
    items = [evidence(gradient_path), evidence("pipeline/analyses/Y45_gradient_prereg_d179.md"),
             evidence("scripts/probe_y45_gradient_attribution_d179.py"), evidence(checks_path), evidence(review_path)]
    candidate["evidence"].extend(items)
    candidate.setdefault("history", []).append(dict(
        observed_at=now, action="native_gradient_conflict_attributed_and_executability_reviewed",
        prior_scientific_status=candidate["scientific_status"], evidence=items))
    candidate.update(
        hypothesis="Native FP/FN partitions distinguish targets at equal scalar IoU; test whether independently learned conditional error evidence improves final confidence after fixed released-backbone training.",
        scientific_status="six_executability_gates_reviewed_not_admitted",
        pending_gates=[], gates_review=evidence(checks_path),
        next_action="Commit exactly seven tested Y45 files; bind real six-gate admission and decision, then canonical GPU1-only dispatch and increasing native-step verification. Preserve all initial conflicts and failure evidence.")
    work.pop("design_cycle", None)
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(gates=checks["gates"], review=evidence(review_path),
                         scientific_status=checks["scientific_status"]), indent=2))
finally:
    writer.close()
