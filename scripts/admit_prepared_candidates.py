"""Review immutable Y52/Y54 evidence and admit to the existing guarded queue."""
import argparse
import ast
from datetime import datetime, timedelta
import json
import math
from pathlib import Path

import yaml
import authorize_prepared_candidates as scoped

ROOT = scoped.ROOT
tick, closure, monitor = scoped.tick, scoped.closure, scoped.monitor
OLD52 = "89ffd8db36cf43ed8cfd75a68c77b4b9"
OLD54 = "fde01739a9864bae9bb342a577235e66"


def load(relative):
    value = tick.load(ROOT / relative)
    if value.get("complete") is not True:
        raise ValueError("Incomplete native evidence: " + relative)
    return value


def analysis(stem, suffix):
    return f"pipeline/analyses/{stem}_{suffix}.json"


def validate_native():
    back_path = analysis("y52_backward_scalar_compat_89ff", OLD52)
    resource_path = analysis("y52_resources_batch_compat_89ff", OLD52)
    review52_path = analysis("y52_native_gate_review", OLD52)
    back, resource, review52 = (load(p) for p in (back_path, resource_path, review52_path))
    assert back["actual_native_sparse_forward_and_backward"] and back["paired_same_forward"]
    assert len(back["rows"]) == 8
    assert all(r["identical_actual_match_indices"] and r["native_backward_finite"] for r in back["rows"])
    trained = [r for r in back["rows"] if r["state"] == "trained_parent_diagnostic_only"]
    initial = [r for r in back["rows"] if r["state"] == "released_init"]
    assert len(trained) == len(initial) == 4
    assert all(r["full_query_gradient_signal"]["max_abs"] > 0 for r in trained)
    assert all(r["full_query_gradient_signal"]["max_abs"] == 0 for r in initial)
    assert all(r["full_query_repeat_noise"]["max_abs"] == 0 for r in back["rows"])
    assert len(back["optimizer_steps"]) == 2 and all(back["nonempty_optimizer_restore"].values())
    assert not back["model_or_checkpoint_weights_saved"]
    assert resource["parameter_delta"] == review52["parameter_delta"] == 0
    assert resource["exact_initial_state_equal"] and review52["inference_within_measured_behavioral_repeats"]
    peak52 = max(back["total_peak_reserved_bytes"], *(r["peak_reserved_bytes"] for r in resource["stress"]))
    assert math.isfinite(peak52) and 0 < peak52 < 24 * 1024 ** 3
    diagnosis_path = analysis("y54_fine_support_native_fde0", OLD54)
    native_path = analysis("y54_native_gates_fde0", OLD54)
    review54_path = analysis("y54_native_gate_review", OLD54)
    diagnosis, native, review54 = (load(p) for p in (diagnosis_path, native_path, review54_path))
    assert diagnosis["actual_sparse_forward_and_FPS"] and diagnosis["exact_native_support_positions_verified"]
    assert diagnosis["native_padding_attention_mass"] > 0 and diagnosis["native_block_padding_effect_max"] > 0
    assert diagnosis["corrected_block_padding_effect_max"] == diagnosis["original_block_repeat_max"] == 0
    assert diagnosis["all_valid_negative_control_indicator_max"] == 0
    assert native["actual_independent_package"] and native["actual_native_sparse_FPS_backward"]
    assert native["parameter_delta"] == 0 and native["expressions"] == 32
    assert len(native["native_backward"]) == 4 and len(native["optimizer_steps"]) == 2
    assert all(native["nonempty_optimizer_restore"].values()) and not native["model_or_checkpoint_weights_saved"]
    for row in native["native_backward"]:
        assert row["finite_backward"] and row["finite_nonzero_indicator_gradients"]
        assert len(row["layers"]) == 6
        for layer in row["layers"]:
            assert layer["finite_attention_gradients"] and layer["valid_source_gradient_max_abs"] > 0
            if layer["padded_fine_supports"]:
                assert layer["padding_source_gradient_max_abs"] == 0
    peak54 = native["native_training_peak_reserved_bytes"]
    assert math.isfinite(peak54) and 0 < peak54 < 24 * 1024 ** 3
    return {
        "AdaRES-Y52": dict(native=back_path, resources=resource_path, review=review52_path,
                           controls=analysis("y52_physical_cpu_v2_4567", "4567f5a09ecb4aa89df8f0c2932652df"),
                           peak=peak52, initial_zero_activity=True),
        "AdaRES-Y54": dict(native=native_path, resources=native_path, review=review54_path,
                           controls=diagnosis_path, peak=peak54, initial_zero_activity=False),
    }


def correct_y52_label(token):
    work = tick.load(ROOT / "pipeline/research_work.json")
    c = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y52")
    old_plan_ref = c["post_training_freeze"]
    closure.artifact(ROOT, old_plan_ref)
    old_plan = load(old_plan_ref["path"])
    assert old_plan["protocol"]["evaluation"].startswith("Y54/")
    queue = monitor.load_queue(str(ROOT))
    task = next(t for t in queue["gpu_queues"]["server-2"] if t["id"] == c["id"])
    assert task["status"] == "waiting" and not task.get("pid") and not task.get("launched")
    assert monitor.load_experiment(str(ROOT), c["id"]) is None
    old_receipt_path = task["admission_receipt"]
    old_receipt = tick.load(ROOT / old_receipt_path)
    plan_path = analysis("AdaRES-Y52_post_training_freeze_v2", token)
    receipt_path = "pipeline/admissions/AdaRES-Y52_v2.json"
    assert not (ROOT / plan_path).exists() and not (ROOT / receipt_path).exists()
    now = datetime.now().astimezone().isoformat()
    plan = json.loads(json.dumps(old_plan))
    plan["protocol"]["evaluation"] = "Y52/W2/B1/Y36 each twice; FIRST primary, preserve independent repeat noise"
    plan.update(clarified_at=now, prior_plan=old_plan_ref,
                evaluated_methods=["AdaRES-Y52", "AdaRES-W2", "AdaRES-B1", "AdaRES-Y36"],
                correction="Pre-training administrative candidate-label correction: copied template said Y54, but the admitted experiment is Y52. All training/evaluation settings, references and decision rules are unchanged; no run or evaluation has started.")
    scoped.lease(token)
    tick.save(ROOT / plan_path, plan)
    receipt = json.loads(json.dumps(old_receipt))
    receipt.update(post_training_plan=scoped.ref(plan_path), clarified_at=now,
                   supersedes=scoped.ref(old_receipt_path))
    receipt["gates"]["constraints"]["evidence"] = [
        scoped.ref(plan_path) if entry == old_plan_ref else entry
        for entry in receipt["gates"]["constraints"]["evidence"]]
    tick.save(ROOT / receipt_path, receipt)
    task["admission_receipt"] = receipt_path
    c.update(admission_receipt=receipt_path, post_training_freeze=scoped.ref(plan_path))
    c["evidence"].extend([scoped.ref(plan_path), scoped.ref(receipt_path)])
    closure.admission(ROOT, task)
    tick.save(ROOT / "pipeline/research_work.json", work)
    monitor.save_queue(str(ROOT), queue)
    closure.validate_work(ROOT, tick.load(ROOT / "pipeline/headless/.tick_context.json"), work["candidates"], queue)
    print(json.dumps(dict(id=c["id"], corrected_candidate_label_only=True,
                          previous_plan_retained=old_plan_ref, plan=scoped.ref(plan_path),
                          receipt=receipt_path, training_launches=0), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--correct-y52-label", action="store_true")
    args = parser.parse_args()
    scoped.lease(args.token)
    lock = monitor.acquire_lock(str(ROOT))
    if lock is None:
        raise ValueError("Project writer is busy")
    try:
        if args.correct_y52_label:
            correct_y52_label(args.token)
            return
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidates = {c["id"]: c for c in work["candidates"]}
        queue = monitor.load_queue(str(ROOT))
        targets = tuple(scoped.SPECS)
        if any(t.get("status") in ("waiting", "claiming", "running")
               for group in queue["gpu_queues"].values() for t in group):
            raise ValueError("Active queue changed; review capacity before admission")
        server = monitor.load_servers(str(ROOT))["server-2"]
        assert server["allowed_gpus"] == [0, 1] and server["max_tasks_per_gpu"] == 1
        head = scoped.git("rev-parse", "HEAD").decode().strip()
        evidence = validate_native()
        freeze54_path = analysis("y54_post_training_freeze", OLD54)
        freeze54 = load(freeze54_path)
        assert freeze54["goal"] == scoped.ref("pipeline/goal.json")
        closure.artifact(ROOT, freeze54["evaluator"])
        bindings = {}
        for candidate_id in targets:
            c = candidates[candidate_id]
            assert c["stage"] == "validation"
            bundle_ref, rows = scoped.checked_bundle(candidate_id, c)
            for row in rows:
                import hashlib
                assert hashlib.sha256(scoped.git("show", head + ":" + row["path"])).hexdigest() == row["sha256"]
                if row["path"].endswith(".py"):
                    for node in ast.walk(ast.parse((ROOT / row["path"]).read_bytes())):
                        if isinstance(node, ast.ImportFrom) and node.module:
                            assert "adares_" not in node.module or scoped.SPECS[candidate_id][0] in node.module
            cfg = yaml.safe_load((ROOT / rows[-1]["path"]).read_text(encoding="utf-8"))
            assert cfg["train"]["epochs"] == 30 and cfg["train"]["seed"] == 1999
            assert cfg["model"]["task_type"] == "res" and cfg["model"]["sampling_module"]["num_proposal"] == 128
            binding_path = f"pipeline/analyses/{candidate_id}_commit_binding_{args.token}.json"
            binding = load(binding_path)
            assert binding["files"] == rows and binding["exact_committed_tested_bytes"]
            assert binding["only_seven_additions"] and binding["unrelated_index_unchanged"]
            scoped.git("merge-base", "--is-ancestor", binding["code_commit"], head)
            auth = load(f"pipeline/authorizations/{candidate_id}_commit.json")
            assert auth["human_message_id"] == scoped.MESSAGE_ID and auth["files"] == rows
            assert scoped.ref(evidence[candidate_id]["review"]) in c["evidence"]
            bindings[candidate_id] = (bundle_ref, binding_path)
            if (ROOT / f"pipeline/admissions/{candidate_id}.json").exists():
                raise ValueError("Admission already exists; do not overwrite or replay")
        if args.check_only:
            print(json.dumps(dict(checked=True, code_commit=head, candidates=evidence,
                                  remote_connectivity="unavailable; no GPU availability or launch claim"), indent=2))
            return
        scoped.lease(args.token)
        assert scoped.git("rev-parse", "HEAD").decode().strip() == head
        now = datetime.now().astimezone()
        network_path = analysis("prepared_candidates_connectivity", args.token)
        tick.save(ROOT / network_path, dict(
            complete=True, recorded_at=now.isoformat(), kind="actual_foreground_tool_observations",
            canonical_preflight=scoped.ref("pipeline/headless/.foreground_monitor.txt"),
            observations=[
                dict(target="server-2 / 192.168.1.163:22", result="connection_timeout", wrapper_exit_code=1),
                dict(target="configured server-02-songcl-gongwang / 222.200.181.37:22",
                     result="connection_timeout", wrapper_exit_code=1),
            ], live_GPU_state=None, identity_confirmed=False, remote_mutations=0,
            interpretation="Connectivity blocks deployment, not measured feasibility. No GPU is inferred free; no training started."))
        freeze52_path = analysis("AdaRES-Y52_post_training_freeze", args.token)
        tick.save(ROOT / freeze52_path, dict(
            complete=True, id="AdaRES-Y52", preregistered_at=now.isoformat(),
            tested_bundle=bindings["AdaRES-Y52"][0], goal=scoped.ref("pipeline/goal.json"),
            design=scoped.ref("pipeline/designs/exp_design_Y52_4567.md"),
            training="Released SPFormer / seed1999 / fixed30 / single GPU / 128 queries / unchanged single-argmax; only tested auxiliary target correction.",
            protocol={**freeze54["protocol"], "evaluation": "Y52/W2/B1/Y36 each twice; FIRST primary, preserve independent repeat noise"},
            comparators=freeze54["comparators"], evaluator=freeze54["evaluator"],
            prior_plan_retained=True, comparator_source=scoped.ref(freeze54_path),
            limitations="Original indicator-regression premise disproved. Released-init auxiliary activity is zero; fixed trained-parent gradients are not Y52 trained efficacy. No validation tuning or diagnostic weights."))
        for gpu, candidate_id in enumerate(targets):
            c = candidates[candidate_id]
            paths = evidence[candidate_id]
            freeze_path = freeze52_path if candidate_id == "AdaRES-Y52" else freeze54_path
            review_path = f"pipeline/analyses/{candidate_id}_committed_admission_{args.token}.md"
            initial = ("Released-init auxiliary activity is zero and retained; four fixed trained-parent batches show a nonzero shared-query intervention, not trained Y52 benefit."
                       if candidate_id == "AdaRES-Y52" else
                       "Native fine-support validity defect has separated padding/positive/negative controls; six fine layers have finite valid gradients and zero invalid-source gradients.")
            (ROOT / review_path).write_text(
                f"# {candidate_id} Committed Admission Review\n\nRecorded {now.isoformat()}; deployment HEAD {head}.\n\n"
                "| Gate | Reviewed Scope |\n|---|---|\n"
                f"| diagnosis / controls | Immutable task-relevant native and matched controls: {paths['controls']} |\n"
                f"| native_backward | {initial} Nonempty model/optimizer/scheduler/RNG restore verified. |\n"
                f"| resources | Zero added parameters; measured native peak {paths['peak']} bytes below 24 GiB. Bounded inputs only, not global Pareto or live availability. |\n"
                "| independence | Seven exact committed files per independent package/config; no cross-variant imports. |\n"
                "| constraints | Current goal hash, fixed30/seed1999/released backbone/128 queries and frozen paired evaluation; no validation tuning. |\n\n"
                "Local commit permission comes from the owning user's continuation. This permission is not itself a scientific PASS or a new training permission; existing project foreground research/admission and server-2 limits still apply.\n\n"
                "Both configured SSH routes timed out. GPU state, current remote bytes and training progress are UNKNOWN. Only waiting admission is recorded; canonical dispatch must freshly verify source marker, GPU occupation/leases and increasing native steps. No remote upload or launch occurred.\n",
                encoding="utf-8")
            decision = monitor.log_decision(
                str(ROOT), "manual_" + candidate_id.lower(),
                candidate_id + "已按本会话继续指示提交精确七文件，是否按既有门禁准入？",
                "六项有范围限制的既有证据及当前提交已复核，加入server-2规范等待队列；连接超时，未部署或启动。",
                "保留原生负面/零活动和训练收益未知；本地提交许可不替代科学审查，单卡单实验固定30不变。",
                review_path + "; " + network_path)
            decision_ref = scoped.ref(Path(decision).relative_to(ROOT).as_posix())
            review_ref = scoped.ref(review_path)
            receipt_path = f"pipeline/admissions/{candidate_id}.json"
            receipt = dict(id=candidate_id, candidate_id=candidate_id,
                config=f"configs_song/model_paper5/{scoped.SPECS[candidate_id][0]}_res.yaml",
                code_commit=head, goal_sha256=scoped.ref("pipeline/goal.json")["sha256"],
                decision=decision_ref, gates={
                    "diagnosis": dict(passed=True, evidence=[scoped.ref(paths["controls"]), review_ref]),
                    "controls": dict(passed=True, evidence=[scoped.ref(paths["controls"]), scoped.ref(paths["review"]), review_ref]),
                    "native_backward": dict(passed=True, evidence=[scoped.ref(paths["native"]), review_ref]),
                    "resources": dict(passed=True, evidence=[scoped.ref(paths["resources"]), scoped.ref(network_path), review_ref]),
                    "independence": dict(passed=True, evidence=[bindings[candidate_id][0], scoped.ref(bindings[candidate_id][1]), review_ref]),
                    "constraints": dict(passed=True, evidence=[scoped.ref("pipeline/goal.json"), scoped.ref(freeze_path), review_ref]),
                }, admitted_at=now.isoformat(), post_training_plan=scoped.ref(freeze_path),
                limitations=initial + " Trained efficacy and novelty unknown; deployment awaits live checks.")
            (ROOT / receipt_path).parent.mkdir(exist_ok=True)
            tick.save(ROOT / receipt_path, receipt)
            task = dict(id=candidate_id, label=candidate_id + " fixed30", config=receipt["config"],
                        status="waiting", gpu=str(gpu), depends_on=[], code_commit=head,
                        admission_receipt=receipt_path, patience=30, min_epochs=30, fixed_epochs=30,
                        requires_exclusive_ownership=True, scientific_candidate_id=candidate_id)
            closure.admission(ROOT, task)
            queue["gpu_queues"]["server-2"].append(task)
            c.update(stage="blocked", blocker_kind="external_resource", required_server="server-2",
                     blocker="server-2 private and configured public SSH routes timed out; current identity, GPU state and remote bytes unknown. Local scoped commit permission is resolved.",
                     retry_after=(now + timedelta(hours=1)).isoformat(),
                     unblock_condition="Server-2 connectivity returns; canonical dispatcher verifies current source marker, authorized GPU occupation/leases and native increasing steps.",
                     next_action="Continue SAME admitted waiting candidate via canonical server-2 dispatcher when reachable; do not ask again for scoped commits, relaunch blindly, or alter the frozen protocol.",
                     code_commit=head, admission_receipt=receipt_path,
                     post_training_freeze=scoped.ref(freeze_path))
            c["evidence"].extend([review_ref, decision_ref, scoped.ref(receipt_path), scoped.ref(network_path), scoped.ref(freeze_path)])
        work["design_cycle"] = dict(stop_reason="external_resource_blocked", attempted_ids=list(targets),
            evidence=[scoped.ref(network_path)], resume_condition="Canonical server-2 connection and safe queued dispatch resume; no additional scoped commit consent is missing.")
        tick.save(ROOT / "pipeline/research_work.json", work)
        monitor.save_queue(str(ROOT), queue)
        context = tick.load(ROOT / "pipeline/headless/.tick_context.json")
        closure.validate_work(ROOT, context, work["candidates"], queue)
        print(json.dumps(dict(admitted=list(targets), code_commit=head, status="waiting", launches=0,
                              connectivity=scoped.ref(network_path)), indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
