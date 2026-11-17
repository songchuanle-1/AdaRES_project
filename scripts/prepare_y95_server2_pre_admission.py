"""Run actual-artifact pending admission guards without GPU use or scientific replay."""
import argparse
import copy
from datetime import datetime
import json
import subprocess
import sys

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y94_deployment import Y94, git
import closure
import y95_server2_pre_admission as checks


def prepare(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y94
        before = {path: ref(path) for path in
                  ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == checks.CID)
        original = copy.deepcopy(candidate)
        others = copy.deepcopy([row for row in work["candidates"] if row["id"] != checks.CID])
        assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "exclusive_gpu_slot"
        assert not candidate.get("native_smoke_result") and not candidate.get("admission_receipt")
        assert not candidate.get("server2_pre_admission_preparation")
        base = ROOT / ("pipeline/checks/y95_server2_pre_admission_" + token)
        output, log = base.with_suffix(".json"), base.with_suffix(".log")
        assert not output.exists() and not log.exists()
        response = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "scripts",
            "-p", "test_y95_server2_pre_admission.py", "-v"],
            cwd=ROOT, capture_output=True, timeout=90)
        log.write_bytes(response.stdout + response.stderr)
        assert response.returncode == 0 and b"Ran 17 tests" in log.read_bytes(), str(log)
        assert b"skipped" not in log.read_bytes()
        pending = checks.pending(work, Y94)
        assert pending[0] == original
        helpers = [ref("scripts/" + name) for name in (
            "y95_server2_pre_admission.py", "test_y95_server2_pre_admission.py",
            "prepare_y95_server2_pre_admission.py")]
        for key in ("CPU_result", "CPU_program", "source_manifest", "native_preparation",
                    "native_smoke_plan_server1", "native_smoke_plan_server2_gpu0"):
            closure.artifact(ROOT, candidate[key])
        parentref = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y93")["post_training_freeze"]
        assert all(ref(path) == value for path, value in before.items())
        assert git("ls-files", "--stage", "-z") == index
        tick.save(output, dict(
            complete=True, token=token, id=checks.CID, prepared_at=datetime.now().astimezone().isoformat(),
            code_commit=checks.COMMIT, working_HEAD=Y94, pending_operation_tests_passed=17,
            scope=dict(server="server-2", gpu=0, power_limit_watts=280.),
            original_candidate_activation_preserved=True, both_original_native_plans_preserved=True,
            original_Y93_Y94_and_all_other_candidates_preserved=True,
            scientific_files_and_native_programs_unchanged=True, CPU_scientific_tests_replayed=False,
            native_GPU_executed=False, native_admission_tests_using_actual_result_executed=False,
            native_result_fixture_created=False, parent_freeze=parentref,
            full_original_training_and_review_in_memory_only=True, formal_freeze_registered=False,
            full_launch_toolchain_prepared=False, admitted=False, queued=False, source_selected=False,
            production_HEAD_index_queue_running_preserved=True,
            next_action="Only at HEADY94 after matching original Y93/server2 natural30/full finite "
                "checkpoint verification/canonical archive and fresh exclusive idle/unreserved authorized "
                "GPU0 execute run_y95_bound_native_y94.py --server server-2. Actual native PASS then "
                "enables read-only y95_server2_pre_admission.py verification. Before source selection "
                "create a NEW tested Y94 additive terminal handoff bound to actual Y95 native evidence; "
                "complete exact source selection/full original freeze/receipt/admission/canonical "
                "dispatch/increasing native counters in that same authorized tick when feasible. "
                "This preparation is not native PASS, registered freeze, full launch toolchain or admission. "
                "The unchanged server1 alternative requires its own matching future execution chain.",
            evidence=helpers + [ref(log), candidate["additive_version_binding"],
                candidate["native_smoke_plan_server1"], candidate["native_smoke_plan_server2_gpu0"],
                parentref, ref(checks.PAUSE)]))
        candidate["server2_pre_admission_preparation"] = ref(output)
        candidate["evidence"].extend(helpers + [ref(output), ref(log)])
        candidate["next_action"] = tick.load(output)["next_action"]
        assert {key: value for key, value in candidate.items() if key not in (
            "server2_pre_admission_preparation", "evidence", "next_action")} == {
                key: value for key, value in original.items() if key not in ("evidence", "next_action")}
        assert [row for row in work["candidates"] if row["id"] != checks.CID] == others
        work["tick_attempt"] = dict(
            tick_started_at=tick.load(ROOT / "pipeline/headless/.tick_model_view.json")["generated_at"],
            candidate_id=checks.CID, kind="implementation_test", result="passed",
            command="python -B scripts/prepare_y95_server2_pre_admission.py --token " + token,
            observation="Seventeen new actual-artifact pending source/private-runtime/280W/protocol/"
                "current-owner guards passed; missing actual native evidence and still-running Y93 "
                "were refused. No GPU/native-result fixture/scientific replay/activation/registered "
                "freeze/admission/queue/launch change.",
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
