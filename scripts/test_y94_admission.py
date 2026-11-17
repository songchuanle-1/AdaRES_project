"""Use only the actual Y94 native geometry result to exercise admission guards."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
import advance_y94 as advance
import test_y94_execution_guards as execution_guards


class Admission(execution_guards.ExecutionGuards):
    def setUp(self):
        work = tick.load(ROOT / "pipeline/research_work.json")
        self.c = copy.deepcopy(next(row for row in work["candidates"] if row["id"] == "AdaRES-Y94"))
        assert self.c.get("native_smoke_result"), "Actual Y94 native evidence is mandatory"
        self.raw = tick.load(ROOT / self.c["native_smoke_result"]["path"])
        plan = tick.load(ROOT / self.c["native_smoke_plan"]["path"])
        self.f = ast.literal_eval(ast.parse(
            (ROOT / plan["program"]["path"]).read_text(encoding="utf-8")).body[0].value)

    def check(self, value=None):
        advance.validate_result(self.raw if value is None else value, self.c, self.f)

    def test_actual_native_result_and_chosen_runtime(self):
        self.check()
        self.assertEqual(self.c["required_server"], "server-1")
        self.assertEqual(self.f["runtime_power_limit_watts"], 350.)

    def test_identity_version_yaml_goal(self):
        for key, value in (("id", "AdaRES-Y93"), ("code_commit", "0" * 40), ("config", "other.yaml"),
                           ("goal_sha256", "0" * 64), ("native_YAML_sha256", "0" * 64)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_native_entry_two_finite_losses_and_steps(self):
        for key, value in (("split", "val"), ("native_entry", "synthetic"), ("steps", 1),
                           ("optimizer_steps", 1), ("losses", [1.]),
                           ("losses", [1., float("nan")]), ("losses", [1., float("inf")])):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_native_data_backward_paths_and_no_validation(self):
        for key in ("native_data_loaded", "backward_finite", "intended_path_observed"):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: False}))
        for key in ("validation_used", "model_weights_saved", "admission"):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: True}))

    def test_geometry_and_all_six_ordered_self_attention_consumers(self):
        for key, value in (("geometry_calls", 0), ("shared_head_calls", 0),
                           ("query_self_stage_calls", [0]), ("attention_stage_calls", [0])):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_per_step_finite_gradients_updates_and_hard_masks(self):
        for key, value in (("all_finite", False), ("query_gradients_finite", False),
                           ("native_hard_masks_preserved", False), ("indicator_tensors", 0),
                           ("geometry_weight_gradient_max", 0.), ("parameter_update_max", float("nan"))):
            changed = copy.deepcopy(self.raw)
            changed["gradient_checks"][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(changed)

    def test_full_position_head_query_formula_and_zero_anchor_required(self):
        for key, value in (("native_positions_verified", False), ("hard_masks_verified", False),
                           ("consumed_by_all_query_self_attentions", False), ("heads", 1),
                           ("reference_query_rows", [0]), ("bias_max", 0.)):
            changed = copy.deepcopy(self.raw)
            changed["geometry_paths"][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(changed)
        changed = copy.deepcopy(self.raw)
        changed["geometry_paths"][0]["bias_max"] = 1.
        with self.assertRaises(AssertionError):
            self.check(changed)

    def test_native_data_and_four_distinct_scenes(self):
        for key, value in (("config_data_root", "/wrong/data"), ("dataset_class", "synthetic")):
            changed = copy.deepcopy(self.raw)
            changed["data_source"][key] = value
            with self.assertRaises(AssertionError):
                self.check(changed)
        changed = copy.deepcopy(self.raw)
        changed["batches"][1] = copy.deepcopy(changed["batches"][0])
        with self.assertRaises(AssertionError):
            self.check(changed)

    def test_native_memory_is_finite_and_within_device(self):
        for value in (0, 24 * 2**30, float("nan")):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, peak_reserved_bytes=value))


if __name__ == "__main__":
    unittest.main()
