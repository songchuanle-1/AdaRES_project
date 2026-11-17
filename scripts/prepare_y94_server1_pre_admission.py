"""Execute new pre-admission boundary tests without scientific replay or GPU use."""
import argparse
import copy
from datetime import datetime
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y93_deployment import Y93, git
import closure
import y94_server1_pre_admission as checks


def prepare(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y93
        before = {path: ref(path) for path in (
            "pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == checks.CID)
        original = copy.deepcopy(candidate)
        assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "exclusive_gpu_slot"
        assert not candidate.get("native_smoke_result") and not candidate.get("admission_receipt")
        assert not candidate.get("server1_pre_admission_preparation")
        protected = {row["id"]: copy.deepcopy(row) for row in work["candidates"]
                     if row["id"] in ("AdaRES-Y95", "AdaRES-Y96", "AdaRES-Y97", "AdaRES-Y98")}
        base = ROOT / ("pipeline/checks/y94_server1_pre_admission_" + token)
        output, log = base.with_suffix(".json"), base.with_suffix(".log")
        assert not output.exists() and not log.exists()
        response = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "scripts",
            "-p", "test_y94_server1_pre_admission.py", "-v"],
            cwd=ROOT, capture_output=True, timeout=90)
        log.write_bytes(response.stdout + response.stderr)
        assert response.returncode == 0 and b"Ran 17 tests" in log.read_bytes(), str(log)
        assert b"skipped" not in log.read_bytes()
        pending = checks.pending(work, Y93)
        assert pending[0] == original
        helper_refs = [ref("scripts/" + name) for name in (
            "y94_server1_pre_admission.py", "test_y94_server1_pre_admission.py",
            "prepare_y94_server1_pre_admission.py")]
        for key in ("CPU_result", "CPU_program", "source_manifest", "native_preparation",
                    "native_smoke_plan_server1", "native_smoke_plan_server2_gpu0"):
            closure.artifact(ROOT, candidate[key])
        assert all(ref(path) == value for path, value in before.items())
        assert git("ls-files", "--stage", "-z") == index
        tick.save(output, dict(
            complete=True, token=token, id=checks.CID, prepared_at=datetime.now().astimezone().isoformat(),
            code_commit=checks.COMMIT, working_HEAD=Y93, pending_operation_tests_passed=17,
            scope=dict(server="server-1", gpu=0, power_limit_watts=350.),
            original_candidate_activation_preserved=True, both_original_native_plans_preserved=True,
            original_Y92_Y93_and_other_preparations_preserved=True,
            scientific_files_and_native_programs_unchanged=True, CPU_scientific_tests_replayed=False,
            native_GPU_executed=False, native_admission_tests_using_actual_result_executed=False,
            native_result_fixture_created=False, parent_freeze=checks.ref(
                next(row for row in work["candidates"] if row["id"] == "AdaRES-Y92")["post_training_freeze"]["path"]),
            full_original_training_and_review_in_memory_only=True, formal_freeze_registered=False,
            full_launch_toolchain_prepared=False, admitted=False, queued=False, source_selected=False,
            production_HEAD_index_queue_running_preserved=True,
            next_action="At HEADY93 only, after exact original Y92 natural30/full finite checkpoint "
                "verification/canonical archive and fresh exclusive idle server1GPU0, execute the unchanged "
                "run_y94_bound_native_y93.py --server server-1. Actual native PASS then enables read-only "
                "y94_server1_pre_admission.py verification. Still create a NEW tested Y93 additive terminal "
                "handoff bound to that actual Y94 native evidence before source selection; finish exact "
                "source selection/full original freeze/receipt/admission/canonical dispatch/increasing "
                "native counters in the same authorized tick when feasible. This preparation is not a "
                "native result, full launch toolchain, registered freeze, admission or launch. Original "
                "server2 alternative remains unchanged and needs its own matching future execution path.",
            evidence=helper_refs + [ref(log), candidate["additive_version_binding"],
                candidate["native_smoke_plan_server1"], candidate["native_smoke_plan_server2_gpu0"],
                ref(checks.PAUSE), ref(checks.GRANT)]))
        candidate["server1_pre_admission_preparation"] = ref(output)
        candidate["evidence"].extend(helper_refs + [ref(output), ref(log)])
        candidate["next_action"] = tick.load(output)["next_action"]
        assert {key: value for key, value in candidate.items() if key not in (
            "server1_pre_admission_preparation", "evidence", "next_action")} == {
                key: value for key, value in original.items() if key not in ("evidence", "next_action")}
        assert all(row == protected[row["id"]] for row in work["candidates"] if row["id"] in protected)
        work["tick_attempt"] = dict(
            tick_started_at=tick.load(ROOT / "pipeline/headless/.tick_model_view.json")["generated_at"],
            candidate_id=checks.CID, kind="implementation_test", result="passed",
            command="python -B scripts/prepare_y94_server1_pre_admission.py --token " + token,
            observation="Seventeen new actual-artifact pre-admission/source/runtime/protocol/owner guards "
                "passed; missing native evidence and still-running Y92 were refused. No GPU/native-result "
                "fixture/scientific replay/activation/registered freeze/admission/queue/launch change.",
            evidence=[ref(output), ref(log)])
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(preparation=ref(output), new_pending_operation_tests=17,
            native_executed=False, formal_freeze_registered=False, admitted=False, launched=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    prepare(parser.parse_args().token)
