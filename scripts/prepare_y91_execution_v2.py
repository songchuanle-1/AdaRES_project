"""Test the exact operational helpers without GPU use or fabricated admission."""
import argparse
from datetime import datetime
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y91_deployment import Y90, Y91
import closure


def main(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        before = {path: ref(path) for path in (
            "pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = closure.git(ROOT, "ls-files", "--stage", "-z")
        assert closure.git(ROOT, "rev-parse", "HEAD") == Y90
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y91")
        assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "exclusive_gpu_slot"
        assert candidate["code_commit"] == Y91 and not candidate.get("native_smoke_result")
        assert not candidate.get("execution_preparation")
        generation = ref("pipeline/checks/y91_execution_helpers_" + token + ".json")
        generated = tick.load(closure.artifact(ROOT, generation))
        for evidence in generated["evidence"] + generated["originals"]:
            closure.artifact(ROOT, evidence)
        output = ROOT / ("pipeline/checks/y91_execution_preparation_v2_" + token + ".json")
        log = output.with_suffix(".log")
        assert not output.exists() and not log.exists()
        response = subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", "scripts",
             "-p", "test_y91_execution_guards_v2.py", "-v"],
            cwd=ROOT, capture_output=True, timeout=90)
        log.write_bytes(response.stdout + response.stderr)
        assert response.returncode == 0 and b"Ran 13 tests" in log.read_bytes()
        assert b"skipped" not in log.read_bytes()
        helpers = [ref("scripts/" + name) for name in (
            "advance_y91.py", "y91_deployment.py", "test_y91_execution_guards.py",
            "test_y91_execution_guards_v2.py", "test_y91_admission.py",
            "prepare_y91_execution.py", "prepare_y91_execution_v2.py")]
        assert all(ref(path) == value for path, value in before.items())
        assert closure.git(ROOT, "ls-files", "--stage", "-z") == index
        tick.save(output, dict(
            complete=True, token=token, id=candidate["id"], code_commit=Y91,
            prepared_at=datetime.now().astimezone().isoformat(), source_resource_protocol_guards=13,
            evidence=helpers + [ref(log), generation, candidate["additive_version_binding"]],
            actual_native_executed=False, admitted=False, production_HEAD_preserved=Y90,
            index_queue_running_preserved=True, signed_originals_unchanged=True,
            fixture_correction=dict(
                found_by="pre-execution source inspection", original_test_executed=False,
                original=ref("scripts/test_y91_execution_guards.py"),
                replacement=ref("scripts/test_y91_execution_guards_v2.py"),
                change="The wrong-candidate negative case uses Y92 instead of the valid Y91 ID.",
                business_guards_and_scientific_files_unchanged=True),
            next_action="After exact original Y89 fixed30 checkpoint verification/archive and fresh "
                        "exclusive idle server-2 GPU0, run the retained bound native plan. Actual native PASS "
                        "then exact source selection, admission and canonical dispatch. No GPU1/Y57 use."))
        candidate["execution_preparation"] = ref(output)
        candidate["evidence"].extend(helpers + [ref(output), ref(log)])
        candidate["next_action"] = tick.load(output)["next_action"]
        work["tick_attempt"] = dict(
            tick_started_at=tick.load(ROOT / "pipeline/headless/.tick_model_view.json")["generated_at"],
            candidate_id=candidate["id"], kind="implementation_test", result="passed",
            command="python -B scripts/prepare_y91_execution_v2.py --token " + token,
            observation="Thirteen source/protocol/resource guards passed; only a pre-execution wrong-ID "
                        "negative fixture corrected in new files. No GPU task, admission or launch.",
            evidence=[ref(output), ref(log)])
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(preparation=ref(output), operational_guards=13,
                              native_executed=False, admitted=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    main(parser.parse_args().token)
