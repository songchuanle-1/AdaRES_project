"""Record exact Y89 archival, Y91 native admission/launch and preserved Y90."""
import argparse
import copy
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from advance_y91 import make_freeze, prepared, validate_result
from run_y91_bound_native import validate_terminal
from y91_deployment import Y89, Y90, Y91, contract, git
import closure


def record(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y91
        before_q, before_r = ref("pipeline/queue.json"), ref("pipeline/running.json")
        work = tick.load(ROOT / "pipeline/research_work.json")
        y91, _, _, frozen = prepared(work)
        assert y91["stage"] == "training"
        raw = tick.load(closure.artifact(ROOT, y91["native_smoke_result"]))
        validate_result(raw, y91, frozen)
        receipt = ref(y91["admission_receipt"])
        assert receipt == ref("pipeline/admissions/AdaRES-Y91.json")
        native = tick.load(closure.artifact(ROOT, y91["native_smoke"]))
        assert native["losses"] == raw["losses"] and native["server"] == "server-2"
        y89 = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y89")
        parent = tick.load(closure.artifact(ROOT, y89["post_training_freeze"]))
        review = tick.load(closure.artifact(ROOT, y91["post_training_freeze"]))
        expected = make_freeze(parent, y89["post_training_freeze"])
        assert review["evaluation"] == expected["evaluation"] and review["training"] == parent["training"]
        assert review["parent_protocol"] == y89["post_training_freeze"]
        assert review["resource_condition"] == expected["resource_condition"]
        done = monitor.load_experiment(str(ROOT), "AdaRES-Y89")
        assert done["status"] == "done_complete" and str(done["gpu"]) == "0"
        terminal = tick.load(closure.artifact(ROOT, done["terminal_evidence"]))
        validate_terminal(terminal, "server-2")
        assert terminal["checkpoints"][0]["best_miou"] == .5036926865577698
        assert "/20261106_122357/" in terminal["checkpoints"][0]["path"]
        archive = tick.load(closure.artifact(ROOT, y89["terminal_archive_binding"]))
        assert archive["only_exact_Y89_archived"] and archive["checkpoint_files_retained"]
        assert archive["remote_mutations"] is False and archive["full_frozen_evaluation_executed"] is False
        assert y89["stage"] == "blocked" and y89["blocker_kind"] == "exclusive_gpu_slot"
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        assert running == {"server-1": {"0": "AdaRES-Y90"}, "server-2": {"0": "AdaRES-Y91"}}
        observations = {}
        for name, cid, version in (("server-1", "AdaRES-Y90", Y90), ("server-2", "AdaRES-Y91", Y91)):
            task = next(row for row in queue["gpu_queues"][name] if row["id"] == cid)
            assert task["status"] == "running" and task["launch_verified"] is True
            assert str(task["gpu"]) == "0" and task["code_commit"] == version and task["pid"] > 0
            if cid == "AdaRES-Y90":
                assert task["pid"] == 2442962
            closure.admission(ROOT, task)
            candidate = next(row for row in work["candidates"] if row["id"] == cid)
            observed = tick.load(closure.artifact(ROOT, candidate["latest_training_observation"]))
            assert observed["token"] == token and observed["native_progress_confirmed"] is True
            assert observed["pid"] == task["pid"] and observed["source_marker"] == version
            age = datetime.now().astimezone() - datetime.fromisoformat(observed["observed_local"])
            assert 0 <= age.total_seconds() < 300, "Collect a fresh labeled native observation"
            observations[cid] = dict(evidence=candidate["latest_training_observation"],
                                     phase=observed["after"]["latest"], pid=task["pid"])
        paused = next(row for row in queue["gpu_queues"]["server-2"] if row["id"] == "AdaRES-Y57")
        assert paused["status"] == "paused" and paused["dispatch_hold"]
        assert paused["resume_requires_user_authorization"] is True
        assert set(contract(Y90, Y91, "AdaRES-Y90")) == {Y90, Y91}
        y90 = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y90")
        handoff = tick.load(closure.artifact(ROOT, y90["terminal_handoff_plan"]))
        assert handoff["CPU_guard_tests_passed"] == 25 and handoff["additive_guard_tests_passed"] == 6
        assert handoff["checkpoint_loaded"] is False and handoff["archived"] is False
        assert handoff["remote_mutations"] is False
        assert handoff["supersedes"] == y90["pre_y91_terminal_handoff_plan"]
        assert handoff["post_training_freeze"] == y90["post_training_freeze"]
        assert handoff["tested_additive_native_evidence"] == y91["native_smoke_result"]
        for dependency in handoff["dependencies"]:
            closure.artifact(ROOT, dependency)
        guards = ref("pipeline/checks/adares-y91_admission_guards_" + token + ".log")
        log = closure.artifact(ROOT, guards).read_bytes()
        assert b"Ran 22 tests" in log and b"skipped" not in log
        selection = ref("pipeline/checks/adares-y91_source_selection_" + token + ".json")
        selected = tick.load(closure.artifact(ROOT, selection))
        assert selected["parent"] == Y90 and selected["target"] == Y91
        assert selected["other_index_entries_preserved"] and selected["working_files_modified"] is False
        dispatch = ref("pipeline/checks/adares-y91_dispatch_" + token + ".json")
        launched = tick.load(closure.artifact(ROOT, dispatch))
        assert launched["task"]["launch_verified"] is True and launched["task"]["status"] == "running"
        assert launched["other_queue_entries_preserved"] is True
        y92 = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y92")
        assert y92["stage"] == "blocked" and y92["blocker_kind"] == "exclusive_gpu_slot"
        assert not y92.get("native_smoke_result") and not y92.get("admission_receipt")
        assert not (ROOT / "pipeline/admissions/AdaRES-Y92.json").exists()
        for key in ("source_manifest", "CPU_result", "CPU_program", "native_preparation",
                    "native_smoke_plan_server1", "native_smoke_plan_server2_gpu0"):
            closure.artifact(ROOT, y92[key])
        output = ROOT / ("pipeline/checks/y91_execution_closeout_" + token + ".json")
        assert not output.exists()
        dependencies = [
            done["terminal_evidence"], y89["terminal_archive_binding"], receipt,
            y91["native_smoke_result"], y91["native_smoke"], y91["post_training_freeze"],
            y91["latest_training_observation"], y90["latest_training_observation"],
            y90["terminal_handoff_plan"], selection, dispatch, guards, ref(__file__)]
        tick.save(output, dict(
            complete=True, token=token, recorded_at=datetime.now().astimezone().isoformat(),
            original_Y89_fixed30_verified_and_archived=True, training_best_not_FIRST=True,
            actual_Y91_native_pass=True, Y91_admitted=True, Y91_native_progress_verified=True,
            original_Y90_process_preserved=True, Y90_explicit_additive_terminal_binding=True,
            Y92_prepared_only=True, Y92_needs_new_tested_same_byte_binding_on_Y91=True,
            GPU1_and_Y57_still_user_paused=True, full_frozen_FIRST_and_cost_preserved=True,
            new_scientific_candidates=0, new_admissions=1, new_training_launches=1,
            exact_original_archives=1, actual_GPU_native_tasks=1, observations=observations,
            efficacy_measured=False, evidence=dependencies))
        y91["evidence"].append(ref(output))
        y92.update(
            blocker="Original Y90/server1 and newly verified Y91/server2 reserve both authorized GPU0 slots. "
                    "Original Y92 source/plans are preserved and still GPU-unexecuted.",
            next_action="Do not replay CPU/version/plan checks or use the stale run_y92_native.py directly. "
                        "Prepare a NEW explicitly tested additive identity/runtime/current-owner binding of "
                        "the SAME seven scientific files/native bodies on Y91. Only after matching Y90/server1 "
                        "or Y91/server2 exact full30 checkpoint verification/archive and fresh exclusive idle "
                        "unreserved GPU0 run the bound native YAML/TRAIN task, then admit/dispatch on actual PASS. "
                        "Preserve GPU1/Y57 pauses and original frozen evaluation.",
            unblock_condition="NEW tested identical-byte Y92 binding on current Y91 source; matching original "
                              "Y90/server1 or Y91/server2 full30 checkpoint/archive and fresh matching idle GPU0.")
        y92["evidence"].append(ref(output))
        tick.save(ROOT / "pipeline/research_work.json", work)
        assert (before_q, before_r) == (ref("pipeline/queue.json"), ref("pipeline/running.json"))
        print(json.dumps(dict(closeout=ref(output), observations=observations,
                              new_admissions=1, launches=1, archives=1)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    record(parser.parse_args().token)
