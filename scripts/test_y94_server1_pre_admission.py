"""New pending-operation tests using actual artifacts, never fake native results."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick, monitor, ref
import y94_server1_pre_admission as checks
from run_y94_bound_native_y93 import runtime
from check_ef_execution import validate_frozen
from y94_contract import validate_result


class PendingOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tick.load(ROOT / "pipeline/research_work.json")
        cls.row = next(row for row in cls.work["candidates"] if row["id"] == checks.CID)
        cls.plan = tick.load(ROOT / cls.row["native_smoke_plan_server1"]["path"])
        cls.source = (ROOT / cls.plan["program"]["path"]).read_text(encoding="utf-8")
        cls.frozen = ast.literal_eval(ast.parse(cls.source).body[0].value)
        cls.owner = next(row for row in cls.work["candidates"] if row["id"] == "AdaRES-Y92")
        cls.parentref = cls.owner["post_training_freeze"]
        cls.parent = tick.load(ROOT / cls.parentref["path"])

    def test_exact_retained_seven_file_delta(self):
        rows = [row.split("\t") for row in checks.git(
            "diff", "--no-renames", "--name-status", checks.Y93, checks.COMMIT).decode().splitlines()]
        checks.validate_delta(rows)
        self.assertEqual(checks.git("rev-parse", checks.COMMIT + "^").decode().strip(), checks.Y93)

    def test_modified_missing_or_unrelated_source_refused(self):
        rows = [["A", path] for path in checks.paths()]
        for changed in (rows[:-1], rows + [["A", "other.py"]], [["M", rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                checks.validate_delta(changed)

    def test_pending_actual_server1_payload_is_read_only(self):
        before = copy.deepcopy(self.work)
        row, plan, program, frozen = checks.pending(self.work, checks.Y93)
        self.assertEqual(row["id"], checks.CID)
        self.assertEqual(plan["program"], ref(program))
        self.assertEqual(frozen, self.frozen)
        self.assertEqual(self.work, before)

    def test_original_candidate_activation_and_both_plans_unchanged(self):
        self.assertEqual(self.row["stage"], "blocked")
        self.assertEqual(self.row["blocker_kind"], "exclusive_gpu_slot")
        self.assertEqual(self.row["required_server"], "server-2")
        self.assertEqual(self.row["native_smoke_plan"], self.row["native_smoke_plan_server2_gpu0"])
        actual = tick.load(ROOT / "pipeline/research_work.json")
        other = next(row for row in actual["candidates"] if row["id"] == checks.CID)
        self.assertEqual(other, self.row)

    def test_stale_HEAD_and_wrong_commit_refused(self):
        with self.assertRaises(AssertionError):
            checks.pending(self.work, checks.Y92)
        changed = copy.deepcopy(self.work)
        row = next(row for row in changed["candidates"] if row["id"] == checks.CID)
        row["code_commit"] = "0" * 40
        with self.assertRaises(AssertionError):
            checks.pending(changed, checks.Y93)

    def test_original_server1_private_runtime_and_power(self):
        server = monitor.load_servers(str(ROOT))["server-1"]
        self.assertEqual(runtime("server-1", self.row, self.frozen, server), 350.)
        self.assertEqual(self.frozen["runtime_path_mappings"], server["runtime_path_mappings"])
        for key, value in (("runtime_physical_gpu", 1), ("runtime_power_limit_watts", 280.),
                           ("runtime_server", "server-2")):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                runtime("server-1", self.row, dict(self.frozen, **{key: value}), server)

    def test_server2_original_alternative_remains_separate(self):
        plan = tick.load(ROOT / self.row["native_smoke_plan_server2_gpu0"]["path"])
        source = (ROOT / plan["program"]["path"]).read_text(encoding="utf-8")
        frozen = ast.literal_eval(ast.parse(source).body[0].value)
        self.assertEqual(frozen["variant"], self.frozen["variant"])
        self.assertEqual(frozen["runtime_server"], "server-2")
        self.assertEqual(frozen["runtime_power_limit_watts"], 280.)
        self.assertNotEqual(frozen["resource_authorization"], self.frozen["resource_authorization"])

    def test_current_goal_pause_and_whitelists(self):
        self.assertEqual(ref("pipeline/goal.json")["sha256"], checks.GOAL)
        self.assertEqual(ref(checks.GRANT)["sha256"], checks.GRANT_SHA)
        self.assertEqual(ref(checks.PAUSE)["sha256"], checks.PAUSE_SHA)
        servers = monitor.load_servers(str(ROOT))
        self.assertEqual(servers["server-1"]["allowed_gpus"], [0])
        self.assertEqual(servers["server-2"]["allowed_gpus"], [0])
        self.assertEqual(servers["server-6"]["allowed_gpus"], [])

    def test_actual_tested_frozen_files(self):
        self.assertEqual(validate_frozen(self.row, self.plan, self.frozen), 7)
        self.assertEqual(self.plan["execution_helper"], ref("scripts/run_y94_bound_native_y93.py"))

    def test_tampered_payload_or_goal_refused(self):
        for field in ("goal", "source"):
            changed = copy.deepcopy(self.frozen)
            if field == "goal":
                changed["goal_sha256"] = "0" * 64
            else:
                changed["variant"]["python_files"]["dec.py"]["base64"] = "Y2hhbmdlZA=="
            with self.subTest(field=field), self.assertRaises(AssertionError):
                validate_frozen(self.row, self.plan, changed)

    def test_missing_native_result_stops_before_activation(self):
        self.assertFalse(self.row.get("native_smoke_result"))
        self.assertFalse(self.row.get("admission_receipt"))
        self.assertFalse((ROOT / "pipeline/admissions/AdaRES-Y94.json").exists())
        with self.assertRaisesRegex(AssertionError, "Actual Y94 native evidence"):
            checks.require_native(self.row, self.frozen)

    def test_actual_CPU_result_cannot_impersonate_native(self):
        cpu = tick.load(ROOT / self.row["CPU_result"]["path"])
        self.assertFalse(cpu["native_smoke_passed"])
        with self.assertRaises((AssertionError, KeyError)):
            validate_result(cpu, self.row, self.frozen)

    def test_full_original_training_and_review_inherited_only_in_memory(self):
        before = copy.deepcopy(self.parent)
        draft = checks.make_freeze(self.parent, self.parentref)
        self.assertEqual(self.parent, before)
        self.assertEqual(draft["id"], checks.CID)
        self.assertEqual(draft["training"], self.parent["training"])
        restored = copy.deepcopy(draft["evaluation"])
        restored["methods"][0] = "AdaRES-Y92"
        self.assertEqual(restored, self.parent["evaluation"])
        self.assertEqual(draft["resource_condition"], self.parent["resource_condition"])
        self.assertEqual(draft["parent_protocol"], self.parentref)

    def test_wrong_protocol_lineage_or_hash_refused(self):
        for key, value in (("id", "AdaRES-Y93"), ("code_commit", "0" * 40), ("complete", False)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                checks.make_freeze(dict(self.parent, **{key: value}), self.parentref)
        with self.assertRaises(AssertionError):
            checks.make_freeze(self.parent, dict(self.parentref, sha256="0" * 64))

    def test_protocol_GPU_or_power_substitution_refused(self):
        for key, value in (("server", "server-2"), ("gpu", 1), ("power_limit_watts", 280.)):
            changed = copy.deepcopy(self.parent)
            changed["resource_condition"][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                checks.make_freeze(changed, self.parentref)

    def test_actual_Y92_not_complete_and_both_owners_reserved(self):
        with self.assertRaisesRegex(AssertionError, "Y92 is not canonically complete"):
            checks.require_completed_owner()
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        self.assertEqual(running, {"server-1": {"0": "AdaRES-Y92"}, "server-2": {"0": "AdaRES-Y93"}})
        for name in ("server-1", "server-2"):
            with self.subTest(server=name), self.assertRaises(AssertionError):
                checks.unreserved(queue, running, name, 0)

    def test_new_helpers_parse_on_original_Python38(self):
        for name in ("y94_server1_pre_admission.py", "test_y94_server1_pre_admission.py",
                     "prepare_y94_server1_pre_admission.py"):
            ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"), feature_version=(3, 8))


if __name__ == "__main__":
    unittest.main()
