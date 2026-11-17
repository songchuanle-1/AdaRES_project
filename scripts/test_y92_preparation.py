"""Require actual visual-side CPU evidence and unchanged original native inputs."""
import ast
import copy
import unittest
from prepare_y92_mask_feature_quadratic import ROOT, PARENT, EDITS, files
from prepare_y60_stage_heads import tick
from y92_contract import validate_cpu, validate_gradients
from y92_observer_core import check_response_rows


class Preparation(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y92')
        self.cpu = tick.load(ROOT / self.c['CPU_result']['path'])

    def test_actual_CPU_evidence(self):
        validate_cpu(self.cpu)

    def test_CPU_cannot_claim_native_or_admission(self):
        for key in ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward',
                    'native_sparse_forward', 'validation_used', 'admission'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: True}))

    def test_anchors_original_evidence_and_causal_bypass_required(self):
        for key in ('decoder_all_stages_anchor_exact', 'decoder_input_gradients_anchor_close',
                    'bypass_restoration_exact', 'intended_initial_masks_changed_initial_scores_unchanged',
                    'shared_visual_features_and_original_hard_gate_preserved'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: False}))

    def test_original_feature_identity_and_all_consumers_required(self):
        for key in ('original_x_mask_output_identity', 'input_undetached',
                    'float64_quadratic_projection_reference', 'shared_module_identity',
                    'all_seven_original_mask_consumers'):
            rows = copy.deepcopy(self.cpu['CPU_projection_paths'])
            rows[1][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                check_response_rows(rows)

    def test_zero_then_positive_learned_response_required(self):
        for index, key, value in ((0, 'weight_nonzero', True), (0, 'delta_max', .1),
                                  (1, 'weight_nonzero', False), (1, 'delta_max', 0.),
                                  (1, 'delta_max', float('nan'))):
            rows = copy.deepcopy(self.cpu['CPU_projection_paths'])
            rows[index][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                check_response_rows(rows)

    def test_missing_reordered_or_duplicate_step_refused(self):
        original = self.cpu['CPU_projection_paths']
        for rows in (original[:-1], list(reversed(original)), [original[0], original[0]]):
            with self.assertRaises(AssertionError):
                check_response_rows(rows)

    def test_actual_gradients_updates_and_consumer_counts_required(self):
        for key, value in (('new_gradient_max', 0.), ('parameter_update_max', float('inf')),
                           ('indicator_tensors', 0), ('input_gradients_finite', False),
                           ('projection_calls', 7), ('original_mask_consumers', 0),
                           ('coarse_gate_consumers', 0)):
            rows = copy.deepcopy(self.cpu['CPU_decoder_steps'])
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_gradients(rows)

    def test_literal_CPU_native_callbacks_identical(self):
        core = (ROOT / self.c['observer_core']['path']).read_text(encoding='utf-8')
        expected = {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(core).body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(len(expected), 13)
        for key in ('CPU_program', 'prepared_native_body'):
            source = (ROOT / self.c[key]['path']).read_text(encoding='utf-8')
            found = {node.name: ast.dump(node, include_attributes=False)
                     for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in expected}
            self.assertEqual(found, expected)
        self.assertIn('projection_calls=2', source)
        self.assertIn('shared_transformed_visual_mask_features_consumed=True', source)
        self.assertNotIn('mask_query_quadratic', source)

    def test_three_edit_reversal_restores_original_W2(self):
        source = files()[2].read_text(encoding='utf-8')
        for old, new in reversed(EDITS):
            self.assertEqual(source.count(new), 1)
            source = source.replace(new, old)
        self.assertEqual(source, (files()[2].parent.parent / PARENT / 'dec.py').read_text(encoding='utf-8'))

    def test_other_native_sources_and_YAML_unchanged(self):
        for path in files()[1:-1]:
            if path.name != 'dec.py':
                self.assertEqual(path.read_bytes(), (path.parent.parent / PARENT / path.name).read_bytes())
        parent = ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')
        self.assertEqual(files()[-1].read_bytes(), parent.read_bytes())


if __name__ == '__main__':
    unittest.main()
