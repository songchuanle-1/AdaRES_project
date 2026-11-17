"""Actual CPU evidence guards and pending native-source checks, not native fixtures."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
from y94_contract import validate_cpu, validate_result


class Preparation(unittest.TestCase):
    def setUp(self):
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        owner = tick.load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")
        self.token = owner["token"]
        self.cpu = tick.load(ROOT / ("pipeline/analyses/y94_cpu_" + self.token + ".json"))
        self.body = (ROOT / ("pipeline/checks/y94_native_" + self.token + "_body.py")).read_text(encoding="utf-8")
        self.assertTrue(view["experiment_first"]["enabled"])

    def test_actual_CPU_evidence(self):
        validate_cpu(self.cpu)

    def test_identity_and_parameter_count_refused(self):
        for key, value in (("id", "AdaRES-Y93"), ("parameters", 1), ("additional_parameters", 0)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: value}))

    def test_component_control_failure_refused(self):
        changed = copy.deepcopy(self.cpu)
        changed["component_controls"][0]["independent_axis_formula"] = False
        with self.assertRaises(AssertionError):
            validate_cpu(changed)

    def test_all_stage_attention_and_hard_masks_required(self):
        for key, value in (("actual_hard_exclusions_preserved", False),
                           ("decoder_all_stages_anchor_exact", False),
                           ("decoder_query_self_consumption_calls", 5), ("decoder_bias_calls", 0)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: value}))

    def test_bypass_and_input_gradient_controls_required(self):
        for key in ("bypassed_geometry_negative_control_exact", "decoder_input_gradients_anchor_close"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: False}))

    def test_two_synthetic_updates_are_finite_and_real(self):
        for key, value in (("parameter_update_max", 0.), ("new_gradient_max", float("nan")),
                           ("input_gradients_finite", False), ("step", 0)):
            changed = copy.deepcopy(self.cpu)
            changed["CPU_decoder_steps"][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(changed)

    def test_CPU_cannot_impersonate_native_evidence(self):
        for key in ("CUDA_initialized", "native_smoke_passed", "native_task_backward",
                    "native_sparse_forward", "validation_used", "admission"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: True}))
        with self.assertRaises((AssertionError, KeyError)):
            validate_result(self.cpu, {"id": "AdaRES-Y94"}, {"variant": {"id": "AdaRES-Y94"}})

    def test_native_body_uses_original_train_and_consumer_guards(self):
        ast.parse(self.body, feature_version=(3, 8))
        self.assertIn("native.train(1,model,FirstTwoBatches()", self.body)
        self.assertIn("model.dec.qqa_layers", self.body)
        self.assertNotIn("model.dec.qia_layers", self.body)
        self.assertIn("assert torch.equal(gate,~neighbors)", self.body)
        self.assertIn("assert torch.equal(kwargs['attn_mask'],expected)", self.body)
        self.assertIn("torch.equal(query_pos,expected_queries)", self.body)

    def test_native_body_keeps_actual_two_updates_and_original_modes(self):
        self.assertIn("optimizer.step=checked_step", self.body)
        self.assertIn("assert len(losses)==len(gradients)==len(batches)==len(modes)==len(geometry_paths)==2", self.body)
        self.assertIn("module.text_encoder.training", self.body)
        self.assertIn("assert all(p.grad is None for p in model.text_encoder.parameters())", self.body)
        self.assertIn("model_weights_saved=False,validation_used=False,admission=False", self.body)

    def test_no_native_result_or_admission_exists(self):
        from prepare_y60_stage_heads import tick
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y94")
        self.assertFalse(candidate.get("native_smoke_result"))
        self.assertFalse(candidate.get("admission_receipt"))
        self.assertFalse((ROOT / "pipeline/admissions/AdaRES-Y94.json").exists())


if __name__ == "__main__":
    unittest.main()
