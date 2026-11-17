"""Check the new exact Y90 terminal binding without remote checkpoint access."""
import argparse
import copy
from datetime import datetime
import io
import json
import unittest
from unittest.mock import patch

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
import y90_terminal_contract as bound
import closure


class TerminalBinding(unittest.TestCase):
    token = None

    def setUp(self):
        self.server = monitor.load_servers(str(ROOT))["server-1"]
        self.task = dict(bound.SPEC, status="running", launch_verified=True, gpu="0")

    def test_exact_runtime_and_contract(self):
        bound.validate_server(self.server)
        contract = bound.build_contract(self.task, self.server)
        self.assertEqual(contract["allowed_markers"], [bound.Y90])
        self.assertEqual(set(contract["deployments"]), {bound.Y90})

    def test_changed_runtime_or_mapping_refused(self):
        for key, value in (
            ("host", "192.0.2.1"), ("expected_hostname", "other"),
            ("python", "/other/python"), ("runtime_library_path", "/other/lib"),
            ("remote_code_root", "/other/code"), ("work_dir", "/other/checkpoints"),
            ("runtime_path_mappings", []), ("allowed_gpus", [0, 1]),
            ("max_tasks_per_gpu", 2),
        ):
            server = copy.deepcopy(self.server)
            server[key] = value
            with self.subTest(key=key), self.assertRaises((AssertionError, ValueError)):
                bound.validate_server(server)

    def test_changed_task_identity_refused(self):
        for key, value in (
            ("id", "AdaRES-Y89"), ("status", "done_complete"),
            ("launch_verified", False), ("gpu", "1"), ("server", "server-2"),
            ("pid", 60578), ("code_commit", "a" * 40),
            ("config", "other.yaml"), ("log", "/other/run.log"),
        ):
            task = dict(self.task, **{key: value})
            with self.subTest(key=key), self.assertRaises(AssertionError):
                bound.build_contract(task, self.server)

    def test_only_exact_original_marker_and_candidate(self):
        bound.validate_marker(bound.Y90, "AdaRES-Y90")
        for marker, candidate in (
            ("5f1c4bde4b682fabd84fdc84a4341ba2c845fe71", "AdaRES-Y90"),
            ("59baf94e9d3e406afa13b2ddc96968cdcc976cae", "AdaRES-Y90"),
            ("a" * 40, "AdaRES-Y90"), (bound.Y90, "AdaRES-Y89"),
        ):
            with self.subTest(marker=marker, candidate=candidate), self.assertRaises(AssertionError):
                bound.validate_marker(marker, candidate)

    def test_future_head_needs_new_explicit_binding(self):
        with patch.object(bound, "git", return_value=b"a" * 40):
            with self.assertRaises(AssertionError):
                bound.build_contract(self.task, self.server)

    def test_original_location_and_readonly_preparation(self):
        location = tick.load(closure.artifact(ROOT, bound.SPEC["location_evidence"]))
        self.assertEqual(location["checkpoint_directory"], bound.CHECKPOINT_DIRECTORY)
        self.assertEqual(location["log"], self.task["log"])
        self.assertIs(location["checkpoint_loaded"], False)
        self.assertIs(location["remote_mutations"], False)
        source = ROOT / ("pipeline/checks/y90_terminal_original_location_" + self.token + ".py")
        program = source.read_text(encoding="utf-8")
        for forbidden in ("import torch", "torch.load", "nvidia-smi"):
            self.assertNotIn(forbidden, program)
        self.assertIn("process.is_dir()", program)
        self.assertIn("P['config'].encode() in args", program)


def check(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == bound.CID)
        assert candidate["stage"] == "training" and not candidate.get("terminal_evidence")
        plan_ref = candidate["terminal_handoff_plan"]
        plan = tick.load(closure.artifact(ROOT, plan_ref))
        assert plan["token"] == token and plan["CPU_guard_tests_passed"] == 25
        assert plan["checkpoint_loaded"] is False and plan["archived"] is False
        helpers_ref = ref("pipeline/checks/y90_terminal_helpers_" + token + ".json")
        helpers = tick.load(closure.artifact(ROOT, helpers_ref))
        preserved = {path: ref(path) for path in (
            "pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        original_helpers = helpers["originals"]
        index = bound.git("ls-files", "--stage", "-z")
        base = ROOT / ("pipeline/checks/y90_terminal_binding_" + token)
        assert not base.with_suffix(".json").exists() and not base.with_suffix(".log").exists()
        TerminalBinding.token = token
        stream = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TerminalBinding)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        base.with_suffix(".log").write_text(stream.getvalue(), encoding="utf-8", newline="\n")
        assert result.wasSuccessful() and result.testsRun == 6 and not result.skipped
        assert all(ref(path) == value for path, value in preserved.items())
        assert bound.git("ls-files", "--stage", "-z") == index
        assert [ref(ROOT / item["path"]) for item in original_helpers] == original_helpers
        for dependency in plan["dependencies"]:
            closure.artifact(ROOT, dependency)
        tick.save(base.with_suffix(".json"), dict(
            complete=True, token=token, id=bound.CID,
            observed_at=datetime.now().astimezone().isoformat(),
            lifecycle_tests_passed=25, binding_tests_passed=6,
            terminal_handoff_plan=plan_ref, helpers=helpers_ref,
            checkpoint_directory=bound.CHECKPOINT_DIRECTORY,
            original_helpers_unchanged=True, main_index_unchanged=True,
            queue_changed=False, remote_mutations=False, checkpoint_loaded=False,
            archived=False, full_frozen_evaluation_executed=False,
            future_HEAD_requires_new_binding=True,
            evidence=[ref(__file__), ref(base.with_suffix(".log")), plan_ref, helpers_ref]))
        candidate["evidence"].append(ref(base.with_suffix(".json")))
        work["tick_attempt"] = dict(
            tick_started_at=tick.load(ROOT / "pipeline/headless/.tick_model_view.json")["generated_at"],
            candidate_id=bound.CID, kind="implementation_test", result="passed",
            command="python -B scripts/check_y90_terminal_binding.py --token " + token,
            observation="Exact Y90 terminal plan:25 lifecycle and6 new source/runtime/owner guards "
                        "passed; no checkpoint load, GPU task, archive, training restart or efficacy claim.",
            evidence=[ref(base.with_suffix(".json")), ref(base.with_suffix(".log"))])
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(binding=ref(base.with_suffix(".json")),
                              lifecycle_tests=25, binding_tests=6,
                              checkpoint_loaded=False, archived=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    check(parser.parse_args().token)
