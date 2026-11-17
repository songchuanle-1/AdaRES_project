"""Exercise Y95 guards only against the actual native result and original protocol."""
import ast
import copy
import unittest

from prepare_y60_stage_heads import ROOT, monitor, ref, tick
from check_ef_execution import validate_frozen
import advance_y95 as advance
import y95_deployment as deploy


class Admission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tick.load(ROOT / "pipeline/research_work.json")
        cls.c, cls.plan, _, cls.f = advance.prepared(cls.work)
        assert cls.c.get("native_smoke_result"), "Actual Y95 native evidence is mandatory"
        cls.raw = tick.load(ROOT / cls.c["native_smoke_result"]["path"])
        parent = next(row for row in cls.work["candidates"] if row["id"] == "AdaRES-Y93")
        cls.parentref = parent["post_training_freeze"]
        cls.parent = tick.load(ROOT / cls.parentref["path"])

    def check(self, value=None):
        advance.validate_result(self.raw if value is None else value, self.c, self.f)

    def test_actual_native_result_and_original_runtime(self):
        self.check()
        self.assertEqual(self.c["required_server"], "server-2")
        self.assertEqual(self.f["runtime_power_limit_watts"], 280.)
        self.assertEqual(self.plan["execution_helper"], ref("scripts/run_y95_bound_native_y94.py"))
        self.assertEqual(validate_frozen(self.c, self.plan, self.f), 7)

    def test_only_actual_native_tested_seven_additions(self):
        self.assertEqual(deploy.git("rev-parse", deploy.Y95 + "^").decode().strip(), deploy.Y94)
        rows = [row.split("\t") for row in deploy.git(
            "diff", "--no-renames", "--name-status", deploy.Y94, deploy.Y95).decode().splitlines()]
        deploy.validate_delta(rows, "AdaRES-Y95")
        for path in deploy.paths("AdaRES-Y95"):
            self.assertEqual((ROOT / path).read_bytes(), deploy.git("show", deploy.Y95 + ":" + path))

    def test_foreign_modified_or_missing_additions_refused(self):
        rows = [["A", path] for path in deploy.paths("AdaRES-Y95")]
        for bad in (rows[:-1], rows + [["A", "other.py"]], [["M", rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, "AdaRES-Y95")

    def test_current_Y94_source_and_unknown_versions(self):
        self.assertEqual(list(deploy.contract(deploy.Y94, deploy.Y95, "AdaRES-Y94")),
                         [deploy.Y94, deploy.Y95])
        for commit, head in ((deploy.Y95, deploy.Y94), ("0" * 40, deploy.Y95),
                             (deploy.Y94, "0" * 40)):
            with self.assertRaises(AssertionError):
                deploy.contract(commit, head, "AdaRES-Y94")

    def test_identity_version_yaml_and_goal(self):
        for key, value in (("id", "AdaRES-Y94"), ("code_commit", "0" * 40),
                           ("config", "other.yaml"), ("goal_sha256", "0" * 64),
                           ("native_YAML_sha256", "0" * 64)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_original_entry_two_finite_steps_and_losses(self):
        for key, value in (("split", "val"), ("native_entry", "synthetic"), ("steps", 1),
                           ("optimizer_steps", 1), ("losses", [1.]),
                           ("losses", [1., float("nan")]), ("losses", [1., float("inf")])):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_native_data_backward_paths_and_no_validation(self):
        for key in ("native_data_loaded", "backward_finite", "intended_path_observed",
                    "static_mask_features_preserved", "original_initial_prediction_preserved",
                    "original_hard_gate_algorithm_preserved", "native_modes_verified"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: False}))
        for key in ("validation_used", "model_weights_saved", "admission"):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: True}))

    def test_all_shared_path_counts(self):
        for key in ("fusion_calls", "original_norm_consumption_calls", "original_ffn_consumption_calls",
                    "shared_head_calls", "query_norm_consumption_calls"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: 0}))

    def test_per_step_original_consumers_gradients_and_updates(self):
        for key, value in (("all_finite", False), ("both_input_gradients_finite", False),
                           ("indicator_tensors", 0), ("indicator_calls", 0), ("fusion_calls", 0),
                           ("coarse_gate_consumers", 0), ("original_norm_consumption_calls", 0),
                           ("original_ffn_consumption_calls", 0), ("query_norm_consumption_calls", 0),
                           ("new_gradient_max", 0.), ("parameter_update_max", float("nan"))):
            changed = copy.deepcopy(self.raw)
            changed["gradient_checks"][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(changed)

    def test_six_ordered_all_query_paths_and_zero_anchor(self):
        for key, value in (("step", 0), ("stage", 0), ("weight_nonzero", False),
                           ("original_self_identity", False), ("original_language_identity", False),
                           ("both_inputs_undetached", False),
                           ("float64_all_query_mean_projection_reference", False), ("fusion_delta_max", 0.)):
            changed = copy.deepcopy(self.raw)
            changed["fusion_paths"][7][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(changed)
        changed = copy.deepcopy(self.raw)
        changed["fusion_paths"][0]["fusion_delta_max"] = 1.
        with self.assertRaises(AssertionError):
            self.check(changed)

    def test_original_data_and_four_distinct_scenes(self):
        for key, value in (("config_data_root", "/wrong/data"), ("dataset_class", "synthetic")):
            changed = copy.deepcopy(self.raw)
            changed["data_source"][key] = value
            with self.assertRaises(AssertionError):
                self.check(changed)
        changed = copy.deepcopy(self.raw)
        changed["batches"][1] = copy.deepcopy(changed["batches"][0])
        with self.assertRaises(AssertionError):
            self.check(changed)

    def test_memory_is_finite_and_within_device(self):
        for value in (0, 24 * 2**30, float("nan")):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, peak_reserved_bytes=value))

    def test_full_original_training_review_cost_exactly_inherited(self):
        before = copy.deepcopy(self.parent)
        frozen = advance.make_freeze(self.parent, self.parentref)
        self.assertEqual(self.parent, before)
        self.assertEqual(frozen["training"], self.parent["training"])
        restored = copy.deepcopy(frozen["evaluation"])
        restored["methods"][0] = "AdaRES-Y93"
        self.assertEqual(restored, self.parent["evaluation"])
        self.assertEqual(frozen["resource_condition"], self.parent["resource_condition"])
        self.assertEqual(frozen["parent_protocol"], self.parentref)

    def test_wrong_protocol_version_or_resource_refused(self):
        for key, value in (("id", "AdaRES-Y94"), ("code_commit", "0" * 40), ("complete", False)):
            with self.assertRaises(AssertionError):
                advance.make_freeze(dict(self.parent, **{key: value}), self.parentref)
        for key, value in (("gpu", 1), ("server", "server-1"), ("power_limit_watts", 350.)):
            changed = copy.deepcopy(self.parent)
            changed["resource_condition"][key] = value
            with self.assertRaises(AssertionError):
                advance.make_freeze(changed, self.parentref)

    def test_goal_pause_whitelists_and_exact_private_runtime(self):
        servers = monitor.load_servers(str(ROOT))
        self.assertEqual(servers["server-1"]["allowed_gpus"], [0])
        self.assertEqual(servers["server-2"]["allowed_gpus"], [0])
        self.assertEqual(servers["server-6"]["allowed_gpus"], [])
        self.assertEqual(ref(advance.GRANT)["sha256"], advance.GRANT_SHA)
        self.assertEqual(ref(advance.PAUSE)["sha256"], advance.PAUSE_SHA)
        for key, value in (("python", "/wrong/python"), ("expected_hostname", "foreign"),
                           ("allowed_gpus", [0, 1])):
            with self.assertRaises(AssertionError):
                advance.runtime("server-2", self.c, self.f, dict(servers["server-2"], **{key: value}))

    def test_busy_foreign_nonfinite_and_reserved_devices_refused(self):
        advance.idle(dict(gpu="0,1,280", compute=""), 280.)
        for gpu, compute in (("nan,0,280", ""), ("0,0,350", ""), ("0,2000,280", ""),
                             ("0,0,280", "99"), ("100,0,280", "")):
            with self.assertRaises(AssertionError):
                advance.idle(dict(gpu=gpu, compute=compute), 280.)
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        advance.unreserved(queue, running, "server-2", 0)
        with self.assertRaises(AssertionError):
            advance.unreserved(queue, running, "server-1", 0)

    def test_original_Y93_real_complete_checkpoint_required(self):
        done = advance.require_completed_owner()
        proof = tick.load(ROOT / done["terminal_evidence"]["path"])
        self.assertEqual(proof["checkpoints"][0]["epoch"], 30)
        advance.validate_terminal(proof, "server-2")

    def test_original_Y94_actual_native_handoff_required(self):
        deploy.require_running_parent_handoff(self.work, self.c)
        changed = copy.deepcopy(self.c)
        changed["native_smoke_result"] = dict(path="missing.json", sha256="0" * 64)
        with self.assertRaises((AssertionError, KeyError)):
            deploy.require_running_parent_handoff(self.work, changed)

    def test_helpers_parse_for_original_Python38(self):
        for name in ("advance_y95.py", "y95_deployment.py", "test_y95_admission.py",
                     "prepare_y94_y95_terminal_binding.py", "y94_y95_terminal_contract.py"):
            ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"), feature_version=(3, 8))


if __name__ == "__main__":
    unittest.main()
