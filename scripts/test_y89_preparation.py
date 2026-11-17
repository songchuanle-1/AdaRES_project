"""Test real CPU evidence, observer corrections and independent source boundaries."""
import ast
import copy
import unittest
from prepare_y89_stage_evidence_norm import ROOT, PARENT, EDITS, files
from prepare_y60_stage_heads import ref, tick
from y89_contract import validate_cpu, validate_gradients
from y89_observer_core_v2 import check_norm_rows
from repair_y89_typed_reference import BEFORE, AFTER


class Preparation(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = next(candidate for candidate in work['candidates'] if candidate['id'] == 'AdaRES-Y89')
        self.cpu = tick.load(ROOT / self.c['CPU_result']['path'])

    def test_actual_CPU_evidence(self):
        validate_cpu(self.cpu)

    def test_CPU_never_native_or_admission(self):
        for key in ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward', 'native_sparse_forward', 'validation_used', 'admission'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: True}))

    def test_original_anchors_and_private_intervention_required(self):
        for key in ('decoder_all_stages_anchor_exact', 'decoder_input_gradients_anchor_close',
                    'aggregate_shared_evidence_norm_gradient_anchor_close',
                    'private_final_intervention_changes_only_final_confidence', 'restoration_exact'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: False}))

    def test_private_norm_order_and_identity_required(self):
        for key in ('original_query_identity', 'original_mask_identity', 'original_mass_identity',
                    'private_norm_identity', 'undetached_evidence', 'independent_competition_reference',
                    'original_epsilon', 'shared_MLP_consumer'):
            rows = copy.deepcopy(self.cpu['CPU_norm_paths'])
            rows[8][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                check_norm_rows(rows)
        with self.assertRaises(AssertionError):
            check_norm_rows(list(reversed(self.cpu['CPU_norm_paths'])))

    def test_initial_and_learned_responses_required(self):
        for index, key, value in ((0, 'affine_nondefault', True), (0, 'affine_delta_max', .1),
                                   (8, 'affine_nondefault', False), (8, 'affine_delta_max', 0.),
                                   (8, 'affine_delta_max', float('nan'))):
            rows = copy.deepcopy(self.cpu['CPU_norm_paths'])
            rows[index][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                check_norm_rows(rows)

    def test_all_fourteen_gradients_and_updates_required(self):
        for key, value in (('norm_gradient_maxima', 0.), ('norm_parameter_updates', float('nan'))):
            rows = copy.deepcopy(self.cpu['CPU_decoder_steps'])
            rows[1][key][4] = value
            with self.assertRaises(AssertionError):
                validate_gradients(rows)
        for key in ('shared_MLP_parameter_tensors', 'shared_query_norm_calls', 'coarse_gate_consumers'):
            rows = copy.deepcopy(self.cpu['CPU_decoder_steps'])
            rows[1][key] = 0
            with self.assertRaises(AssertionError):
                validate_gradients(rows)

    def test_literal_CPU_native_callback_identity(self):
        source = (ROOT / self.c['observer_core']['path']).read_text(encoding='utf-8')
        expected = {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(len(expected), 10)
        for key in ('CPU_program', 'prepared_native_body'):
            source = (ROOT / self.c[key]['path']).read_text(encoding='utf-8')
            found = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name in expected}
            self.assertEqual(found, expected)

    def test_dtype_reference_fix_preserves_tolerance_and_negative_evidence(self):
        repair = tick.load(ROOT / self.c['observer_reference_repair']['path'])
        old = (ROOT / repair['original_core']['path']).read_text(encoding='utf-8')
        new = (ROOT / repair['updated_core']['path']).read_text(encoding='utf-8')
        self.assertEqual(new.replace(AFTER, BEFORE), old)
        self.assertEqual(repair['tolerance_unchanged'], dict(rtol=1e-5, atol=1e-6))
        self.assertFalse(repair['model_and_YAML_revision'])
        self.assertTrue(repair['original_failure_preserved'])
        diagnosis = tick.load(ROOT / repair['diagnosis']['path'])
        self.assertTrue(diagnosis['same_dtype_reference_exact'])
        self.assertFalse(diagnosis['original_guard_passed'])
        self.assertTrue(diagnosis['original_guard_repeated_unchanged'])
        self.assertEqual(ref(repair['original_CPU_failure']['path']), repair['original_CPU_failure'])

    def test_exact_reversal_restores_W2(self):
        source = files()[2].read_text(encoding='utf-8')
        for old, new in reversed(EDITS):
            if old.startswith('        pred_indis'):
                old, new = '\n' + old, '\n' + new
            self.assertEqual(source.count(new), 1)
            source = source.replace(new, old)
        self.assertTrue(source.startswith('import copy\n'))
        self.assertEqual(source[len('import copy\n'):],
                         (files()[2].parent.parent / PARENT / 'dec.py').read_text(encoding='utf-8'))

    def test_all_other_native_sources_and_YAML_identical(self):
        for path in files()[1:-1]:
            if path.name != 'dec.py':
                self.assertEqual(path.read_bytes(), (path.parent.parent / PARENT / path.name).read_bytes())
        parent = ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')
        self.assertEqual(files()[-1].read_bytes(), parent.read_bytes())


if __name__ == '__main__':
    unittest.main()
