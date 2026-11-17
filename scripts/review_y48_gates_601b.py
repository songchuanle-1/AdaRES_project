"""Review observed executability, retaining both failed diagnostic harnesses."""

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
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 300


def evidence(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


def load(stem):
    path = "pipeline/analyses/" + stem + "_" + args.token + ".json"
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    assert value["complete"] and value["id"] == "AdaRES-Y48"
    return value, evidence(path)


joint, joint_ref = load("y48_joint_native_601b")
cpu, cpu_ref = load("y48_target_cpu_601b")
native, native_ref = load("y48_target_native_601b")
production, production_ref = load("y48_production_v3_601b")
bundle, bundle_ref = load("y48_source_bundle_601b")
assert joint["actual_native_mixed_nonempty_groups"] == 31
assert cpu["CPU_cases"] == 32 and native["native_backward_executed"]
assert native["native_scenes"] == 8 and native["changed_queries"] == 52
assert production["stripped_AST_exact_parent"] and production["configuration_byte_identical_parent"]
assert production["CPU_cases"] == 32 and production["actual_tools_train_steps"] == 2
assert sum(row["new_positive_queries"] for row in production["released_init_closure_records"]) > 0
assert production["model_optimizer_scheduler_restore_exact"]
assert production["cached_physical_parent_decoder_equal"]
assert production["parameters"] == 164629637 and production["added_parameters"] == 0
assert max(row["peak_reserved_gib"] for row in production["released_init_native_stress"]) < 24
assert bundle["physical_independence"] and bundle["remote_byte_identity"]
assert len(bundle["files"]) == 7
for row in bundle["files"]:
    assert evidence(row["path"]) == row
design_ref = evidence("pipeline/designs/exp_design_Y48_601b.md")
gate_refs = {
    "diagnosis": [joint_ref, evidence("pipeline/analyses/Y48_target_closure_prereg_601b.md")],
    "controls": [cpu_ref, native_ref, production_ref],
    "native_backward": [native_ref, production_ref],
    "resources": [production_ref],
    "independence": [bundle_ref, production_ref],
    "constraints": [evidence("pipeline/goal.json"), design_ref],
}
failures = [
    evidence("pipeline/analyses/y48_build_byte_identity_repair_601b_" + args.token + ".json"),
    evidence("pipeline/analyses/y48_production_harness_repair_601b_" + args.token + ".json"),
    evidence("pipeline/analyses/y48_production_buffer_repair_601b_" + args.token + ".json"),
]
review = """# SAME Y48 Executability Review

| Gate | Result | Actual evidence | Limits |
|---|---|---|---|
| diagnosis | PASS for falsifiable experiment | Joint native8scenes/32expressions,31 mixed nonempty identical-mask groups with equal point confusion | not hidden-state identity, necessity or efficacy |
| controls | PASS intended target operation | CPU32 original native CE, empty/positive/query-permutation controls; native52 changed queries; independent production CE/gradient and parent-default equality | no population or global-loader bitwise conclusion |
| native_backward | PASS native path/update/restore | trained-W2 diagnostic8scenes; released SPFormer actual tools.train2steps, indicator updates, full numeric model/optimizer/scheduler restoration | exact two HF metadata buffers checked physically; no other mismatch permitted |
| resources | PASS fixed feasibility | added0 parameters; native largest-pair stress finite/reserved<24GiB; actual serial predict readings | released-init cost, not trained Pareto |
| independence | PASS independent tested bytes | six new physical files/unique config, no cross-variant imports; stripped AST parent equality/local-remote hashes | no novelty claim |
| constraints | PASS fixed protocol | released SPFormer/seed1999/fixed30/128queries/argmax1/server2GPU1 only | no validation tuning or diagnostic-weight deployment |

## Failures Retained
The initial local/native byte guard failed before writing a package. Captured
native parent bytes and six-file AST equality resolved the binding, not a lowered
gate. Production harness v1 failed inherited two-scene predict shape handling;
v2 serial prediction then failed strict loading of the two known RoBERTa metadata
buffers. V3 only repairs the diagnostic convention and explicitly verifies the
physical buffer values plus every model/optimizer/scheduler state. All original
sources, logs, temporary checkpoints and preregistrations remain. Model code and
the proposed intervention did not change during these harness repairs.

## Scientific Limits And Frozen Evaluation
Native one-to-many matching is preserved. OR may overweight poor masks, and hard
mask equivalence need not imply equal latent states or benefit. Native graph and
update activity permits a falsifiable Phase1 test, not confidence in positive
mIoU, no-harm, superiority, necessity, novelty or Phase2 approval.
Use exp_design_Y48_601b.md unchanged: released SPFormer/seed1999/fixed30/singleGPU;
Y48/W2/B1 full9508/141 twice EACH/FIRST, scene-bootstrap10000seed1999/TRAIN size
cuts1487/3325, CI-upper<0 subgroup harm review, repeat/metadata/cost limitations.
Only exactly tested seven files may be committed and receipt-bound before
canonical GPU1 dispatch. Verify increasing native steps, not merely a PID.
"""
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    review_path = "pipeline/analyses/Y48_six_gate_review_601b.md"
    with (ROOT / review_path).open("x", encoding="utf-8") as stream:
        stream.write(review)
    checks = dict(
        complete=True, id="AdaRES-Y48", observed_at_local=datetime.now().astimezone().isoformat(),
        gates={key: True for key in gate_refs}, evidence=gate_refs,
        review=evidence(review_path), preserved_failures=failures,
        scientific_status="executable_for_falsifiable_training_not_efficacy")
    path = "pipeline/analyses/y48_gate_checks_601b_" + args.token + ".json"
    with (ROOT / path).open("x", encoding="utf-8") as stream:
        json.dump(checks, stream, indent=2)
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y48")
    assert candidate["stage"] == "validation"
    candidate["evidence"].extend([
        production_ref, bundle_ref, design_ref, evidence(path), evidence(review_path), *failures])
    candidate.update(
        pending_gates=[], gates_review=evidence(path),
        config="configs_song/model_paper5/adares_y48_mask_equivalence_res.yaml",
        scientific_status="six_executability_gates_reviewed_not_admitted",
        next_action="Commit exactly tested seven Y48 files; bind decision/goal/six gates; canonical GPU1-only dispatch then verify increasing native steps.")
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(gates=checks["gates"], review=checks["review"]), indent=2))
finally:
    writer.close()
