"""Fail-closed guards against the actual original Y96 native result."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, monitor, ref, tick
from check_ef_execution import validate_frozen
import advance_y96 as advance
import y96_deployment as deploy


class Admission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tick.load(ROOT / "pipeline/research_work.json")
        cls.c, cls.plan, _, cls.f = advance.prepared(cls.work)
        assert cls.c.get("native_smoke_result"), "Actual native result required"
        cls.raw = tick.load(ROOT / cls.c["native_smoke_result"]["path"])
        parent = next(row for row in cls.work["candidates"] if row["id"] == "AdaRES-Y94")
        cls.parentref = parent["post_training_freeze"]
        cls.parent = tick.load(ROOT / cls.parentref["path"])

    def check(self, value=None):
        advance.validate_result(self.raw if value is None else value, self.c, self.f)

    def test_actual_result_runtime_and_source(self):
        self.check()
        self.assertEqual(self.f["runtime_server"], "server-1")
        self.assertEqual(self.f["runtime_power_limit_watts"], 350.)
        self.assertEqual(validate_frozen(self.c, self.plan, self.f), 7)
        self.assertEqual(deploy.git("rev-parse", deploy.Y96 + "^").decode().strip(), deploy.Y95)
        deploy.contract(deploy.Y95, deploy.Y96, "AdaRES-Y95")

    def test_wrong_identity_version_goal_and_yaml_refused(self):
        for key, value in (("id", "other"), ("code_commit", "0" * 40), ("config", "other.yaml"),
                           ("goal_sha256", "0" * 64), ("native_YAML_sha256", "0" * 64)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_original_train_two_steps_and_finite_losses(self):
        for key, value in (("split", "val"), ("native_entry", "synthetic"), ("steps", 1),
                           ("optimizer_steps", 1), ("losses", [1.]), ("losses", [1., float("nan")])):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_observed_paths_backward_and_no_validation(self):
        for key in ("complete", "native_data_loaded", "backward_finite", "intended_path_observed",
                    "static_mask_features_preserved", "original_hard_gate_algorithm_preserved",
                    "native_modes_verified"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: False}))
        for key in ("validation_used", "model_weights_saved", "admission"):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: True}))

    def test_original_generator_and_shared_consumers(self):
        for key in ("initialization_calls", "generator_layer_calls", "coarse_gate_consumers",
                    "shared_head_calls", "query_norm_consumption_calls"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: 0}))

    def test_per_step_gradients_and_updates(self):
        for key, value in (("all_finite", False), ("both_input_gradients_finite", False),
                           ("indicator_tensors", 0), ("indicator_calls", 0),
                           ("generator_layer_calls", 0), ("new_gradient_max", 0.),
                           ("parameter_update_max", float("nan"))):
            changed = copy.deepcopy(self.raw)
            changed["gradient_checks"][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(changed)

    def test_data_and_four_distinct_scenes(self):
        changed = copy.deepcopy(self.raw)
        changed["data_source"]["config_data_root"] = "/wrong/data"
        with self.assertRaises(AssertionError):
            self.check(changed)
        changed = copy.deepcopy(self.raw)
        changed["batches"][1] = copy.deepcopy(changed["batches"][0])
        with self.assertRaises(AssertionError):
            self.check(changed)

    def test_device_memory_budget(self):
        for value in (0, 24 * 2**30, float("nan")):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, peak_reserved_bytes=value))

    def test_full_original_server1_protocol_unchanged(self):
        before = copy.deepcopy(self.parent)
        freeze = advance.make_freeze(self.parent, self.parentref)
        self.assertEqual(self.parent, before)
        self.assertEqual(freeze["training"], self.parent["training"])
        restored = copy.deepcopy(freeze["evaluation"])
        restored["methods"][0] = "AdaRES-Y94"
        self.assertEqual(restored, self.parent["evaluation"])
        self.assertEqual(freeze["resource_condition"], self.parent["resource_condition"])
        self.assertEqual(freeze["resource_authorization"], ref(advance.GRANT))

    def test_wrong_parent_resource_refused(self):
        for key, value in (("id", "other"), ("code_commit", "0" * 40), ("complete", False)):
            with self.assertRaises(AssertionError):
                advance.make_freeze(dict(self.parent, **{key: value}), self.parentref)
        changed = copy.deepcopy(self.parent)
        changed["resource_condition"]["server"] = "server-2"
        with self.assertRaises(AssertionError):
            advance.make_freeze(changed, self.parentref)

    def test_pause_whitelists_and_private_runtime(self):
        servers = monitor.load_servers(str(ROOT))
        self.assertEqual(servers["server-1"]["allowed_gpus"], [0])
        self.assertEqual(servers["server-2"]["allowed_gpus"], [0])
        self.assertEqual(servers["server-6"]["allowed_gpus"], [])
        self.assertEqual(ref(advance.GRANT)["sha256"], advance.GRANT_SHA)
        self.assertEqual(ref(advance.PAUSE)["sha256"], advance.PAUSE_SHA)
        for key, value in (("python", "/wrong/python"), ("expected_hostname", "foreign"),
                           ("allowed_gpus", [0, 1])):
            with self.assertRaises(AssertionError):
                advance.runtime("server-1", self.c, self.f, dict(servers["server-1"], **{key: value}))

    def test_busy_foreign_and_reserved_devices(self):
        advance.idle(dict(gpu="0,1,350", compute=""), 350.)
        for gpu, compute in (("nan,0,350", ""), ("0,0,280", ""), ("0,2000,350", ""),
                             ("0,0,350", "99"), ("100,0,350", "")):
            with self.assertRaises(AssertionError):
                advance.idle(dict(gpu=gpu, compute=compute), 350.)
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        advance.unreserved(queue, running, "server-1", 0)
        with self.assertRaises(AssertionError):
            advance.unreserved(queue, running, "server-2", 0)

    def test_original_complete_checkpoint_and_other_live_handoff(self):
        advance.require_completed_owner()
        deploy.require_running_parent_handoff(self.work, self.c)
        changed = copy.deepcopy(self.c)
        changed["native_smoke_result"] = dict(path="missing.json", sha256="0" * 64)
        with self.assertRaises(AssertionError):
            deploy.require_running_parent_handoff(self.work, changed)

    def test_python38_parse(self):
        for name in ("advance_y96.py", "y96_deployment.py", "y95_y96_terminal_contract.py",
                     "prepare_y95_y96_terminal_binding.py", "test_y96_admission.py"):
            ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"), feature_version=(3, 8))
