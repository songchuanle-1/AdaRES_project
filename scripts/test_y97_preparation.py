"""Guard actual token-input evidence, full unchanged source and pending GPU status."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
from prepare_y97_language_quadratic_input import files, EDITS, PARENT, NAMES
from y97_contract import validate_cpu, validate_result


class Preparation(unittest.TestCase):
    def setUp(self):
        self.token = tick.load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")["token"]
        self.cpu = tick.load(ROOT / ("pipeline/analyses/y97_cpu_" + self.token + ".json"))
        self.body = (ROOT / ("pipeline/checks/y97_native_" + self.token + "_body.py")).read_text(encoding="utf-8")

    def test_actual_CPU_evidence(self):
        validate_cpu(self.cpu)

    def test_identity_and_parameter_counts_refused(self):
        for key, value in (("id", "AdaRES-Y96"), ("parameters", 1), ("additional_parameters", 0)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: value}))

    def test_scalar_quadratic_sign_token_locality_no_RNG_required(self):
        for key in ("independent_scalar_raw_token_quadratic_projection", "quadratic_sign_symmetry",
                    "raw_intervention_is_token_local", "no_RNG", "token_batch_input_output_channel_permutations"):
            changed = copy.deepcopy(self.cpu)
            changed["component_controls"][0][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(changed)

    def test_anchors_bypass_and_original_language_consumers_required(self):
        for key in ("decoder_all_stages_anchor_exact", "decoder_input_gradients_anchor_close",
                    "exact_bypass_restoration", "original_raw_tokens_and_projection_consumers_verified",
                    "original_language_norm_sampling_and_first_self_consumers_verified"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: False}))

    def test_two_finite_updates_gradients_frozen_encoder_and_consumers_required(self):
        for key, value in (("parameter_update_max", 0.), ("new_gradient_max", float("nan")),
                           ("decoder_input_gradients_finite", False), ("original_language_norm_calls", 0),
                           ("original_sampling_consumers", 0), ("frozen_text_encoder_preserved", False)):
            changed = copy.deepcopy(self.cpu)
            changed["CPU_decoder_steps"][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(changed)

    def test_raw_full_channels_original_norm_and_sampling_identity_required(self):
        for key, value in (("step", 0), ("weight_nonzero", False), ("language_delta_max", 0.),
                           ("full768_channels_retained", False), ("original_language_norm_consumes_adapter", False),
                           ("original_sampling_consumes_normalized_language", False)):
            changed = copy.deepcopy(self.cpu)
            changed["CPU_language_paths"][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(changed)

    def test_CPU_cannot_impersonate_native_PASS(self):
        for key in ("CUDA_initialized", "native_smoke_passed", "native_task_backward",
                    "native_sparse_forward", "validation_used", "admission"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: True}))
        with self.assertRaises((AssertionError, KeyError)):
            validate_result(self.cpu, {"id": "AdaRES-Y97"}, {"variant": {"id": "AdaRES-Y97"}})

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
        core = (ROOT / "scripts/y97_observer_core.py").read_text(encoding="utf-8")
        cpu = (ROOT / ("pipeline/checks/y97_cpu_" + self.token + "_program.py")).read_text(encoding="utf-8")
        self.assertEqual(cpu.count(core), 1)
        self.assertEqual(self.body.count(core), 1)
        ast.parse(self.body, feature_version=(3, 8))
        for text in ("native.train(1, model, FirstTwoBatches()", "optimizer.step = checked_step",
                     "module.text_encoder.training", "original_projection_calls=2", "state['raw_expected'] = args[3]"):
            self.assertIn(text, self.body)

    def test_no_native_result_or_admission_and_earlier_candidates_preserved(self):
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidates = {row["id"]: row for row in work["candidates"]}
        self.assertFalse(candidates["AdaRES-Y97"].get("native_smoke_result"))
        self.assertFalse(candidates["AdaRES-Y97"].get("admission_receipt"))
        self.assertFalse((ROOT / "pipeline/admissions/AdaRES-Y97.json").exists())
        for cid in ("AdaRES-Y94", "AdaRES-Y95", "AdaRES-Y96"):
            self.assertEqual(candidates[cid]["stage"], "blocked")
            self.assertFalse(candidates[cid].get("native_smoke_result"))


if __name__ == "__main__":
    unittest.main()
