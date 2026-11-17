"""Guard actual CPU evidence and exact independent source, without GPU fixtures."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
from prepare_y95_query_set_mean import files, MODULE, PARENT, NAMES
from y95_contract import validate_cpu, validate_result


class Preparation(unittest.TestCase):
    def setUp(self):
        self.token = tick.load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")["token"]
        self.cpu = tick.load(ROOT / ("pipeline/analyses/y95_cpu_" + self.token + ".json"))
        self.body = (ROOT / ("pipeline/checks/y95_native_" + self.token + "_body.py")).read_text(encoding="utf-8")

    def test_actual_CPU_evidence(self):
        validate_cpu(self.cpu)

    def test_identity_and_parameter_counts_refused(self):
        for key, value in (("id", "AdaRES-Y94"), ("parameters", 1), ("additional_parameters", 0)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: value}))

    def test_independent_scalar_and_peer_causality_required(self):
        for key in ("independent_all_query_scalar_mean_projection", "changed_peer_affects_unchanged_focal_query",
                    "query_replication_invariant", "zero_mean_set_has_no_skip"):
            changed = copy.deepcopy(self.cpu)
            changed["component_controls"][0][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(changed)

    def test_anchors_bypass_and_original_consumers_required(self):
        for key in ("decoder_all_stages_anchor_exact", "decoder_input_gradients_anchor_close",
                    "bypassed_fusion_negative_control_exact", "static_mask_and_original_hard_gate_preserved"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: False}))

    def test_two_finite_updates_and_six_consumers_required(self):
        for key, value in (("parameter_update_max", 0.), ("new_gradient_max", float("nan")),
                           ("both_input_gradients_finite", False), ("fusion_calls", 5), ("coarse_gate_consumers", 0)):
            changed = copy.deepcopy(self.cpu)
            changed["CPU_decoder_steps"][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(changed)

    def test_global_context_paths_cannot_be_missing_or_reordered(self):
        for key, value in (("stage", 0), ("weight_nonzero", False),
                           ("float64_all_query_mean_projection_reference", False), ("fusion_delta_max", 0.)):
            changed = copy.deepcopy(self.cpu)
            changed["CPU_fusion_paths"][-1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(changed)

    def test_CPU_cannot_impersonate_native_PASS(self):
        for key in ("CUDA_initialized", "native_smoke_passed", "native_task_backward",
                    "native_sparse_forward", "validation_used", "admission"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: True}))
        with self.assertRaises((AssertionError, KeyError)):
            validate_result(self.cpu, {"id": "AdaRES-Y95"}, {"variant": {"id": "AdaRES-Y95"}})

    def test_exact_independent_W2_decoder_edits_only(self):
        parent = ROOT / "AdaRES/ipdn/model_paper5" / PARENT
        source = (parent / "dec.py").read_text(encoding="utf-8")
        source = source.replace("class DEC(nn.Module):", MODULE + "class DEC(nn.Module):")
        source = source.replace("        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)",
            "        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n"
            "        self.query_set_skip = QuerySetMeanSkip(d_model)")
        source = source.replace("            query = self.norm_qqa(query_self + query_qla)",
            "            query = self.norm_qqa(self.query_set_skip(query_self, query_qla))")
        target = files()[0].parent
        self.assertEqual(source, (target / "dec.py").read_text(encoding="utf-8"))
        for name in NAMES:
            if name not in ("__init__.py", "dec.py"):
                self.assertEqual((parent / name).read_bytes(), (target / name).read_bytes())
        self.assertEqual(files()[-1].read_bytes(),
            (ROOT / "AdaRES/configs_song/model_paper5" / (PARENT + "_res.yaml")).read_bytes())

    def test_exact_CPU_executed_callbacks_in_native_body(self):
        core = (ROOT / "scripts/y95_observer_core.py").read_text(encoding="utf-8")
        cpu = (ROOT / ("pipeline/checks/y95_cpu_" + self.token + "_program.py")).read_text(encoding="utf-8")
        self.assertEqual(cpu.count(core), 1)
        self.assertEqual(self.body.count(core), 1)
        ast.parse(self.body, feature_version=(3, 8))
        for text in ("native.train(1,model,FirstTwoBatches()", "optimizer.step=checked_step",
                     "module.text_encoder.training", "original_norm_consumption_calls=12",
                     "args[1].double().mean(1, keepdim=True)"):
            self.assertIn(text, self.body)

    def test_no_native_result_or_admission_exists(self):
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y95")
        self.assertFalse(candidate.get("native_smoke_result"))
        self.assertFalse(candidate.get("admission_receipt"))
        self.assertFalse((ROOT / "pipeline/admissions/AdaRES-Y95.json").exists())


if __name__ == "__main__":
    unittest.main()
