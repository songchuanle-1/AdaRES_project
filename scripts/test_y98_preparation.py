"""Guard actual full-appearance evidence without inventing native success."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
from prepare_y98_appearance_mask_skip import files, EDITS, PARENT, NAMES
from y98_contract import validate_cpu, validate_result


class Preparation(unittest.TestCase):
    def setUp(self):
        self.token = tick.load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")["token"]
        self.cpu = tick.load(ROOT / ("pipeline/analyses/y98_cpu_" + self.token + ".json"))
        self.body = (ROOT / ("pipeline/checks/y98_native_" + self.token + "_body.py")).read_text(encoding="utf-8")

    def test_actual_CPU_evidence(self):
        validate_cpu(self.cpu)

    def test_identity_and_parameter_counts_refused(self):
        for key, value in (("id", "AdaRES-Y96"), ("parameters", 1), ("additional_parameters", 0)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: value}))

    def test_scalar_projection_row_locality_sign_no_RNG_required(self):
        for key in ("independent_scalar_full_appearance_projection", "appearance_sign_reversal",
                    "appearance_intervention_is_row_local", "no_RNG", "row_input_output_channel_permutations"):
            changed = copy.deepcopy(self.cpu)
            changed["component_controls"][0][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(changed)

    def test_anchors_bypass_and_original_mask_consumers_required(self):
        for key in ("decoder_all_stages_anchor_exact", "decoder_input_gradients_anchor_close",
                    "exact_bypass_restoration", "original_appearance_and_base_mask_consumers_verified",
                    "all_seven_static_mask_consumers_verified"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: False}))

    def test_two_updates_gradients_frozen_encoder_original_paths_required(self):
        for key, value in (("parameter_update_max", 0.), ("new_gradient_max", float("nan")),
                           ("decoder_input_gradients_finite", False), ("original_visual_parameter_tensors", 0),
                           ("base_adapted_and_original_appearance_gradients_finite", False),
                           ("frozen_text_encoder_preserved", False)):
            changed = copy.deepcopy(self.cpu)
            changed["CPU_decoder_steps"][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(changed)

    def test_full_appearance_original_batching_and_static_reuse_required(self):
        for key, value in (("step", 0), ("weight_nonzero", False), ("mask_feature_delta_max", 0.),
                           ("original_full1024_appearance_identity", False),
                           ("original_batching_consumes_skip", False),
                           ("all_seven_predictions_reuse_same_features", False)):
            changed = copy.deepcopy(self.cpu)
            changed["CPU_appearance_mask_paths"][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(changed)

    def test_CPU_cannot_impersonate_native_PASS(self):
        for key in ("CUDA_initialized", "native_smoke_passed", "native_task_backward",
                    "native_sparse_forward", "validation_used", "admission"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: True}))
        with self.assertRaises((AssertionError, KeyError)):
            validate_result(self.cpu, {"id": "AdaRES-Y98"}, {"variant": {"id": "AdaRES-Y98"}})

    def test_exact_independent_W2_decoder_edits_only(self):
        parent = ROOT / "AdaRES/ipdn/model_paper5" / PARENT
        source = (parent / "dec.py").read_text(encoding="utf-8")
        for old, new in EDITS:
            self.assertEqual(source.count(old), 1)
            source = source.replace(old, new)
        target = files()[0].parent
        self.assertEqual(source, (target / "dec.py").read_text(encoding="utf-8"))
        for name in NAMES:
            if name not in ("__init__.py", "dec.py"):
                self.assertEqual((parent / name).read_bytes(), (target / name).read_bytes())
        self.assertEqual(files()[-1].read_bytes(),
            (ROOT / "AdaRES/configs_song/model_paper5" / (PARENT + "_res.yaml")).read_bytes())

    def test_exact_CPU_executed_callbacks_in_pending_native_body(self):
        core = (ROOT / "scripts/y98_observer_core.py").read_text(encoding="utf-8")
        cpu = (ROOT / ("pipeline/checks/y98_cpu_" + self.token + "_program.py")).read_text(encoding="utf-8")
        self.assertEqual(cpu.count(core), 1)
        self.assertEqual(self.body.count(core), 1)
        ast.parse(self.body, feature_version=(3, 8))
        for text in ("native.train(1, model, FirstTwoBatches()", "optimizer.step = checked_step",
                     "module.text_encoder.training", "original_appearance_projection_calls=2",
                     "state['raw_appearance_expected'] = args[6]"):
            self.assertIn(text, self.body)
        self.assertIn("assert torch.equal(parts(changed)[0]['scores'], parts(baseline)[0]['scores'])", cpu)

    def test_no_native_result_or_admission_and_earlier_candidates_preserved(self):
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidates = {row["id"]: row for row in work["candidates"]}
        self.assertFalse(candidates["AdaRES-Y98"].get("native_smoke_result"))
        self.assertFalse(candidates["AdaRES-Y98"].get("admission_receipt"))
        self.assertFalse((ROOT / "pipeline/admissions/AdaRES-Y98.json").exists())
        for cid in ("AdaRES-Y94", "AdaRES-Y95", "AdaRES-Y96"):
            self.assertEqual(candidates[cid]["stage"], "blocked")
            self.assertFalse(candidates[cid].get("native_smoke_result"))
        self.assertEqual(candidates["AdaRES-Y97"]["blocker_kind"], "numerical_control_failure")
        self.assertEqual(candidates["AdaRES-Y97"]["revision_history"][0]["result"], "failed")


if __name__ == "__main__":
    unittest.main()
