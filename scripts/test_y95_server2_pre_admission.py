"""Actual pending Y95 artifacts, not fabricated native PASS or terminal evidence."""
import ast
import copy
import unittest

from prepare_y60_stage_heads import ROOT, tick, monitor, ref
import y95_server2_pre_admission as checks
from run_y95_bound_native_y94 import runtime
from check_ef_execution import validate_frozen
from y95_contract import validate_result


class PendingOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tick.load(ROOT / "pipeline/research_work.json")
        cls.row = next(row for row in cls.work["candidates"] if row["id"] == checks.CID)
        cls.plan = tick.load(ROOT / cls.row["native_smoke_plan_server2_gpu0"]["path"])
        source = (ROOT / cls.plan["program"]["path"]).read_text(encoding="utf-8")
        cls.frozen = ast.literal_eval(ast.parse(source).body[0].value)
        cls.owner = next(row for row in cls.work["candidates"] if row["id"] == "AdaRES-Y93")
        cls.parentref = cls.owner["post_training_freeze"]
        cls.parent = tick.load(ROOT / cls.parentref["path"])

    def test_exact_retained_seven_file_delta(self):
        rows = [row.split("\t") for row in checks.git(
            "diff", "--no-renames", "--name-status", checks.Y94, checks.COMMIT).decode().splitlines()]
        checks.validate_delta(rows)
        self.assertEqual(checks.git("rev-parse", checks.COMMIT + "^").decode().strip(), checks.Y94)

    def test_missing_modified_or_unrelated_source_refused(self):
        rows = [["A", path] for path in checks.paths()]
        for changed in (rows[:-1], rows + [["A", "other.py"]], [["M", rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                checks.validate_delta(changed)

    def test_actual_pending_payload_is_read_only(self):
        before = copy.deepcopy(self.work)
        row, plan, program, frozen = checks.pending(self.work, checks.Y94)
        self.assertEqual(row["id"], checks.CID)
        self.assertEqual(plan["program"], ref(program))
        self.assertEqual(frozen, self.frozen)
        self.assertEqual(self.work, before)

    def test_candidate_activation_and_both_plans_unchanged(self):
        self.assertEqual(self.row["stage"], "blocked")
        self.assertEqual(self.row["blocker_kind"], "exclusive_gpu_slot")
        self.assertEqual(self.row["required_server"], "server-2")
        self.assertEqual(self.row["native_smoke_plan"], self.row["native_smoke_plan_server2_gpu0"])
        actual = tick.load(ROOT / "pipeline/research_work.json")
        self.assertEqual(next(row for row in actual["candidates"] if row["id"] == checks.CID), self.row)

    def test_stale_HEAD_and_wrong_commit_refused(self):
        with self.assertRaises(AssertionError):
            checks.pending(self.work, checks.Y93)
        changed = copy.deepcopy(self.work)
        next(row for row in changed["candidates"] if row["id"] == checks.CID)["code_commit"] = "0" * 40
        with self.assertRaises(AssertionError):
            checks.pending(changed, checks.Y94)

    def test_original_server2_private_runtime_and_power(self):
        server = monitor.load_servers(str(ROOT))["server-2"]
        self.assertEqual(runtime("server-2", self.row, self.frozen, server), 280.)
        self.assertEqual(server["python"], "/home/songcl/.conda/envs/adares_project/bin/python")
        for key, value in (("runtime_physical_gpu", 1), ("runtime_power_limit_watts", 350.),
                           ("runtime_server", "server-1")):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                runtime("server-2", self.row, dict(self.frozen, **{key: value}), server)
        for key, value in (("python", "/wrong/python"), ("expected_hostname", "foreign"),
                           ("allowed_gpus", [0, 1])):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                runtime("server-2", self.row, self.frozen, dict(server, **{key: value}))

    def test_original_server1_alternative_is_separate(self):
        plan = tick.load(ROOT / self.row["native_smoke_plan_server1"]["path"])
        source = (ROOT / plan["program"]["path"]).read_text(encoding="utf-8")
        frozen = ast.literal_eval(ast.parse(source).body[0].value)
        self.assertEqual(frozen["variant"], self.frozen["variant"])
        self.assertEqual(frozen["runtime_server"], "server-1")
        self.assertEqual(frozen["runtime_power_limit_watts"], 350.)
        self.assertNotEqual(frozen["resource_authorization"], self.frozen["resource_authorization"])
        self.assertEqual(frozen["runtime_path_mappings"],
                         monitor.load_servers(str(ROOT))["server-1"]["runtime_path_mappings"])

    def test_goal_pause_and_whitelists(self):
        self.assertEqual(ref("pipeline/goal.json")["sha256"], checks.GOAL)
        self.assertEqual(ref(checks.GRANT)["sha256"], checks.GRANT_SHA)
        self.assertEqual(ref(checks.PAUSE)["sha256"], checks.PAUSE_SHA)
        servers = monitor.load_servers(str(ROOT))
        self.assertEqual(servers["server-1"]["allowed_gpus"], [0])
        self.assertEqual(servers["server-2"]["allowed_gpus"], [0])
        self.assertEqual(servers["server-6"]["allowed_gpus"], [])

    def test_actual_frozen_sources_and_bound_helper(self):
        self.assertEqual(validate_frozen(self.row, self.plan, self.frozen), 7)
        self.assertEqual(self.plan["execution_helper"], ref("scripts/run_y95_bound_native_y94.py"))

    def test_tampered_source_or_goal_refused(self):
        for field in ("goal", "source"):
            changed = copy.deepcopy(self.frozen)
            if field == "goal":
                changed["goal_sha256"] = "0" * 64
            else:
                changed["variant"]["python_files"]["dec.py"]["base64"] = "Y2hhbmdlZA=="
            with self.subTest(field=field), self.assertRaises(AssertionError):
                validate_frozen(self.row, self.plan, changed)

    def test_missing_native_result_refused_before_activation(self):
        self.assertFalse(self.row.get("native_smoke_result"))
        self.assertFalse(self.row.get("admission_receipt"))
        self.assertFalse((ROOT / "pipeline/admissions/AdaRES-Y95.json").exists())
        with self.assertRaisesRegex(AssertionError, "Actual Y95 native evidence"):
            checks.require_native(self.row, self.frozen)

    def test_actual_CPU_result_is_not_native(self):
        cpu = tick.load(ROOT / self.row["CPU_result"]["path"])
        self.assertFalse(cpu["native_smoke_passed"])
        with self.assertRaises((AssertionError, KeyError)):
            validate_result(cpu, self.row, self.frozen)

    def test_complete_original_protocol_inherited_only_in_memory(self):
        before = copy.deepcopy(self.parent)
        draft = checks.make_freeze(self.parent, self.parentref)
        self.assertEqual(self.parent, before)
        self.assertEqual(draft["id"], checks.CID)
        self.assertEqual(draft["training"], self.parent["training"])
        restored = copy.deepcopy(draft["evaluation"])
        restored["methods"][0] = "AdaRES-Y93"
        self.assertEqual(restored, self.parent["evaluation"])
        self.assertEqual(draft["resource_condition"], self.parent["resource_condition"])
        self.assertEqual(draft["parent_protocol"], self.parentref)

    def test_wrong_protocol_lineage_hash_or_bytes_refused(self):
        for key, value in (("id", "AdaRES-Y94"), ("code_commit", "0" * 40), ("complete", False)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                checks.make_freeze(dict(self.parent, **{key: value}), self.parentref)
        with self.assertRaises(AssertionError):
            checks.make_freeze(self.parent, dict(self.parentref, sha256="0" * 64))
        changed = copy.deepcopy(self.parent)
        changed["evaluation"]["repeats"] = 1
        with self.assertRaises(AssertionError):
            checks.make_freeze(changed, self.parentref)

    def test_protocol_GPU_server_or_power_substitution_refused(self):
        for key, value in (("server", "server-1"), ("gpu", 1), ("power_limit_watts", 350.)):
            changed = copy.deepcopy(self.parent)
            changed["resource_condition"][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                checks.make_freeze(changed, self.parentref)

    def test_current_Y93_incomplete_and_both_GPU0_reserved(self):
        with self.assertRaisesRegex(AssertionError, "Y93 is not canonically complete"):
            checks.require_completed_owner()
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        self.assertEqual(running, {"server-1": {"0": "AdaRES-Y94"}, "server-2": {"0": "AdaRES-Y93"}})
        for name in ("server-1", "server-2"):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                checks.unreserved(queue, running, name, 0)

    def test_helpers_parse_on_original_Python38(self):
        for name in ("y95_server2_pre_admission.py", "test_y95_server2_pre_admission.py",
                     "prepare_y95_server2_pre_admission.py"):
            ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"), feature_version=(3, 8))


if __name__ == "__main__":
    unittest.main()
