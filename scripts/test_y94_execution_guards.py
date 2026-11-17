"""Source, resource and protocol checks on the actual native-tested Y94 edge."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, monitor, ref, tick
from check_ef_execution import validate_frozen
from run_y94_bound_native_y93 import runtime
import y94_deployment as deploy
import advance_y94 as advance


class ExecutionGuards(unittest.TestCase):
    def candidate(self):
        return next(row for row in tick.load(ROOT / "pipeline/research_work.json")["candidates"]
                    if row["id"] == "AdaRES-Y94")

    def parent(self):
        row = next(row for row in tick.load(ROOT / "pipeline/research_work.json")["candidates"]
                   if row["id"] == "AdaRES-Y92")
        return tick.load(ROOT / row["post_training_freeze"]["path"]), row["post_training_freeze"]

    def test_only_actual_native_tested_seven_additions(self):
        self.assertEqual(deploy.git("rev-parse", deploy.Y94 + "^").decode().strip(), deploy.Y93)
        rows = [row.split("\t") for row in deploy.git(
            "diff", "--no-renames", "--name-status", deploy.Y93, deploy.Y94).decode().splitlines()]
        deploy.validate_delta(rows, "AdaRES-Y94")
        for path in deploy.paths("AdaRES-Y94"):
            self.assertEqual((ROOT / path).read_bytes(), deploy.git("show", deploy.Y94 + ":" + path))

    def test_modified_or_unrelated_additions_refused(self):
        rows = [["A", path] for path in deploy.paths("AdaRES-Y94")]
        for bad in (rows[:-1], rows + [["A", "other.py"]], [["M", rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, "AdaRES-Y94")
        with self.assertRaises(AssertionError):
            deploy.validate_delta(rows, "AdaRES-Y95")

    def test_running_Y93_sources_and_unknown_heads(self):
        self.assertEqual(list(deploy.contract(deploy.Y93, deploy.Y94, "AdaRES-Y93")),
                         [deploy.Y93, deploy.Y94])
        for commit, head in ((deploy.Y94, deploy.Y93), ("0" * 40, deploy.Y94),
                             (deploy.Y93, "0" * 40)):
            with self.assertRaises(AssertionError):
                deploy.contract(commit, head, "AdaRES-Y93")

    def test_actual_chosen_native_runtime_and_unchanged_alternative(self):
        work = tick.load(ROOT / "pipeline/research_work.json")
        row, _, _, frozen = advance.prepared(work)
        self.assertEqual(row["required_server"], "server-1")
        self.assertEqual(frozen["runtime_power_limit_watts"], 350.)
        for name, key in (("server-1", "native_smoke_plan_server1"),
                          ("server-2", "native_smoke_plan_server2_gpu0")):
            plan = tick.load(ROOT / row[key]["path"])
            source = (ROOT / plan["program"]["path"]).read_text(encoding="utf-8")
            payload = ast.literal_eval(ast.parse(source).body[0].value)
            self.assertEqual(validate_frozen(row, plan, payload), 7)
            runtime(name, row, payload, monitor.load_servers(str(ROOT))[name])

    def test_goal_pause_whitelists_preserved(self):
        servers = monitor.load_servers(str(ROOT))
        self.assertEqual(servers["server-1"]["allowed_gpus"], [0])
        self.assertEqual(servers["server-2"]["allowed_gpus"], [0])
        self.assertEqual(servers["server-6"]["allowed_gpus"], [])
        self.assertEqual(ref("pipeline/goal.json")["sha256"],
            "4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a")
        self.assertEqual(ref(advance.PAUSE)["sha256"], advance.PAUSE_SHA)
        self.assertEqual(ref(advance.GRANT)["sha256"], advance.GRANT_SHA)

    def test_original_full_training_review_and_cost_inherited_exactly(self):
        parent, parentref = self.parent()
        before = copy.deepcopy(parent)
        frozen = advance.make_freeze(parent, parentref)
        self.assertEqual(parent, before)
        self.assertEqual(frozen["training"], parent["training"])
        restored = copy.deepcopy(frozen["evaluation"])
        restored["methods"][0] = "AdaRES-Y92"
        self.assertEqual(restored, parent["evaluation"])
        self.assertEqual(frozen["resource_condition"]["server"], "server-1")
        self.assertEqual(frozen["resource_condition"]["power_limit_watts"], 350.)
        self.assertEqual(frozen["parent_protocol"], parentref)

    def test_wrong_original_protocol_or_device_refused(self):
        parent, parentref = self.parent()
        for key, value in (("id", "AdaRES-Y93"), ("code_commit", "0" * 40)):
            with self.assertRaises(AssertionError):
                advance.make_freeze(dict(parent, **{key: value}), parentref)
        for key, value in (("gpu", 1), ("server", "server-2"), ("power_limit_watts", 280.)):
            changed = copy.deepcopy(parent)
            changed["resource_condition"][key] = value
            with self.assertRaises(AssertionError):
                advance.make_freeze(changed, parentref)

    def test_busy_foreign_nonfinite_power_or_reservation_refused(self):
        advance.idle(dict(gpu="0,1,350", compute=""), 350.)
        for gpu, compute in (("nan,0,350", ""), ("0,0,280", ""), ("0,2000,350", ""),
                             ("0,0,350", "99"), ("100,0,350", "")):
            with self.assertRaises(AssertionError):
                advance.idle(dict(gpu=gpu, compute=compute), 350.)
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        advance.unreserved(queue, running, "server-1", 0)
        with self.assertRaises(AssertionError):
            advance.unreserved(queue, running, "server-2", 0)

    def test_original_Y92_has_real_complete_checkpoint_proof(self):
        done = monitor.load_experiment(str(ROOT), "AdaRES-Y92")
        self.assertEqual(done["status"], "done_complete")
        proof = tick.load(ROOT / done["terminal_evidence"]["path"])
        advance.validate_terminal(proof, "server-1")
        self.assertTrue(proof["process_absent"])
        self.assertEqual(proof["checkpoints"][0]["epoch"], 30)

    def test_helpers_parse_and_selection_requires_new_running_handoff(self):
        for name in ("advance_y94.py", "y94_deployment.py", "test_y94_admission.py"):
            ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"), feature_version=(3, 8))
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = copy.deepcopy(self.candidate())
        candidate["native_smoke_result"] = {"path": "pipeline/checks/not_actual_y94.json",
                                          "sha256": "0" * 64}
        with self.assertRaises((AssertionError, KeyError)):
            deploy.require_running_parent_handoff(work, candidate)


if __name__ == "__main__":
    unittest.main()
