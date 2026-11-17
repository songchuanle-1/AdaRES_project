"""Attach real bounded test artifacts to the same current candidate IDs."""

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
parser.add_argument("mode", choices=(
    "initial", "cache_cpu", "cache_native", "target_cpu", "target_native",
    "frozen", "cost", "review_y45"))
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 60


def evidence(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


def result(stem, cid):
    path = "pipeline/analyses/" + stem + "_" + args.token + ".json"
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    assert value["complete"] and value["id"] == cid
    return value, evidence(path)


writer = monitor.acquire_lock(str(ROOT))
assert writer is not None, "Current probe still owns writer; do not race its output"
try:
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidates = {row["id"]: row for row in work["candidates"]}
    if args.mode == "initial":
        y46, y48 = candidates["AdaRES-Y46"], candidates["AdaRES-Y48"]
        native, native_ref = result("y46_native_gate_601b", "AdaRES-Y46")
        joint, joint_ref = result("y48_joint_native_601b", "AdaRES-Y48")
        assert native["native_backward_executed"] and native["native_scenes"] == 8
        assert joint["actual_native_mixed_nonempty_groups"] == 31
        prereg, prereg_ref = result("y46_cache_revision_prereg_601b", "AdaRES-Y46")
        assert not y46.get("revision_history")
        y46["revision_history"] = [{
            key: prereg[key] for key in (
                "failure_scope", "reason", "change", "prediction", "preregistered_at")
        }]
        y46["revision_history"][0].update(
            revision=1, failure_evidence=[prereg["failure_evidence"]],
            result="pending", evidence=[prereg_ref])
        y46["evidence"].extend([native_ref, prereg_ref])
        y46.update(
            stage="implementation", failure_scope="implementation",
            scientific_status="native_backward_observed_component_cost_revision_pending",
            next_action="Execute preregistered exact-cache CPU controls then SAME native TRAIN8 cached backward/cost; afterward actual tools.train/released-init controls/independent package and all gates.")
        target_prereg = evidence("pipeline/analyses/Y48_target_closure_prereg_601b.md")
        y48["evidence"].extend([
            joint_ref, target_prereg,
            evidence("scripts/probe_y48_target_closure_601b.py")])
        y48.update(
            stage="implementation",
            scientific_status="native_mixed_mask_labels_observed_final_target_closure_pending",
            next_action="Execute preregistered native-loss CPU32 closure controls and SAME TRAIN8 final-target closure/full backward; then released-init/tools.train/production/resource/independence gates.")
        y48["target_closure_registered_at"] = datetime.now().astimezone().isoformat()
        y48["actual_native_mixed_nonempty_groups"] = 31
        changed = [y46["id"], y48["id"]]
    elif args.mode == "review_y45":
        comparison, comparison_ref = result("y45_clustered_601b", "AdaRES-Y45")
        cost, cost_ref = result("y45_trained_cost_601b", "AdaRES-Y45")
        assert not comparison["target_reached"]
        assert not any(comparison["frozen_harm_triggered_groups"].values())
        candidate = candidates["AdaRES-Y45"]
        assert candidate["stage"] == "validation"
        candidate["evidence"].extend([
            comparison_ref,
            evidence("pipeline/analyses/AdaRES-Y45_frozen_review_601b.md")])
        candidate.update(
            stage="complete", scientific_status="complete_inconclusive",
            failure_scope="inconclusive",
            next_action="No promotion or validation-selected patch. Preserve complete frozen realization; broader mechanism remains unresolved, not falsified.")
        candidate.setdefault("history", []).append(dict(
            observed_at=datetime.now().astimezone().isoformat(),
            action="fixed30_frozen_full_repeats_scene_subgroup_cost_review_complete_inconclusive"))
        changed = [candidate["id"]]
    else:
        bindings = {
            "cache_cpu": ("AdaRES-Y46", "y46_cached_cpu_601b"),
            "cache_native": ("AdaRES-Y46", "y46_cached_native_601b"),
            "target_cpu": ("AdaRES-Y48", "y48_target_cpu_601b"),
            "target_native": ("AdaRES-Y48", "y48_target_native_601b"),
            "frozen": ("AdaRES-Y45", "y45_frozen_601b"),
            "cost": ("AdaRES-Y45", "y45_trained_cost_601b"),
        }
        cid, stem = bindings[args.mode]
        value, ref = result(stem, cid)
        candidate = candidates[cid]
        candidate["evidence"].append(ref)
        if args.mode == "cache_cpu":
            assert value["cache_all_exact"] and value["native_expressions"] == 32
            candidate["revision_history"][0]["evidence"].append(ref)
            candidate["next_action"] = "Run SAME registered cached native TRAIN8 backward/paired-cost; CPU rows/weights/groups exact, not native optimizer or admission."
        elif args.mode == "cache_native":
            assert value["native_backward_executed"] and value["native_scenes"] == 8
            candidate["revision_history"][0]["evidence"].append(ref)
            candidate["revision_history"][0]["result"] = "passed"
            candidate.update(
                stage="validation",
                scientific_status="exact_cache_and_native_path_observed_remaining_gates_pending",
                next_action="Test actual tools.train/released-init optimizer continuation, matched controls and independent production package/resources before any admission.")
        elif args.mode == "target_cpu":
            assert value["CPU_cases"] == 32 and not value["native_backward_executed"]
            candidate["next_action"] = "Run SAME frozen final-target closure TRAIN8 full native loss/backward; CPU controls are not native efficacy."
        elif args.mode == "target_native":
            assert value["native_backward_executed"] and value["native_scenes"] == 8
            candidate.update(
                stage="validation",
                scientific_status="native_final_target_closure_observed_remaining_gates_pending",
                next_action="Execute released-init actual optimizer/control/resource checks, create independent production package and six-gate review before admission.")
        elif args.mode == "frozen":
            assert value["newly_executed_full_passes"] == 6
            candidate["next_action"] = "Complete frozen scene/subgroup comparison and trained paired-cost review; no favorable-repeat selection or retuning."
        else:
            candidate["next_action"] = "Review SAME frozen full/repeat/scene/subgroup/cost evidence; classify this realization honestly, no automatic hypothesis rejection or promotion."
        changed = [cid]
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(
        complete=True, mode=args.mode, changed_ids=changed,
        observed_at_local=datetime.now().astimezone().isoformat(),
        admission_or_training_launches=0), indent=2))
finally:
    writer.close()
