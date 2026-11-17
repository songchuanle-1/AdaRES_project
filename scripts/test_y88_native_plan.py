"""Actual CPU evidence and independent negative controls; never GPU PASS fixtures."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
from y88_contract import validate_cpu, validate_gradients
from y88_observer_core import check_response_rows


class NativePlan(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = next(candidate for candidate in work['candidates'] if candidate['id'] == 'AdaRES-Y88')
        result = self.c.get('CPU_result')
        assert result, 'Actual CPU result binding required'
        self.cpu = tick.load(ROOT / result['path'])

    def test_actual_CPU(self):
        validate_cpu(self.cpu)

    def test_no_GPU_or_admission_claim(self):
        for key in ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward', 'native_sparse_forward', 'admission'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: True}))

    def test_original_paths_and_anchors(self):
        for key in ('decoder_all_stages_anchor_exact', 'decoder_input_gradients_anchor_close',
                    'original_score_and_mask_consumers_verified', 'bypass_restoration_exact'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{key: False}))

    def test_zero_and_nonzero_response(self):
        for index, key, value in ((0, 'delta_max', .1), (8, 'delta_max', 0.),
                                   (0, 'weight_nonzero', True), (8, 'weight_nonzero', False)):
            rows = copy.deepcopy(self.cpu['CPU_projection_paths'])
            rows[index][key] = value
            with self.assertRaises(AssertionError):
                check_response_rows(rows)

    def test_response_finiteness_and_order(self):
        for value in (float('nan'), float('inf'), -1.):
            rows = copy.deepcopy(self.cpu['CPU_projection_paths'])
            rows[8]['delta_max'] = value
            with self.assertRaises(AssertionError):
                check_response_rows(rows)
        for rows in (self.cpu['CPU_projection_paths'][:-1], list(reversed(self.cpu['CPU_projection_paths']))):
            with self.assertRaises(AssertionError):
                check_response_rows(rows)

    def test_exact_identity_and_consumer_flags(self):
        for key in ('original_normalized_query_identity', 'input_undetached',
                    'float64_quadratic_projection_reference', 'original_mask_consumer'):
            rows = copy.deepcopy(self.cpu['CPU_projection_paths'])
            rows[8][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                check_response_rows(rows)

    def test_gradients_updates_and_counts(self):
        for key, value in (('new_gradient_max', 0.), ('parameter_update_max', float('nan')),
                           ('input_gradients_finite', False), ('original_mask_consumers', 0),
                           ('coarse_gate_consumers', 0), ('step', 0)):
            rows = copy.deepcopy(self.cpu['CPU_decoder_steps'])
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_gradients(rows)

    def test_literal_CPU_native_callback_identity(self):
        source = (ROOT / 'scripts/y88_observer_core.py').read_text(encoding='utf-8')
        functions = {node.name: ast.dump(node, include_attributes=False)
                     for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        assert len(functions) == 11
        for key in ('CPU_program', 'prepared_native_body'):
            path = ROOT / self.c[key]['path']
            found = {node.name: ast.dump(node, include_attributes=False) for node in
                     ast.parse(path.read_text(encoding='utf-8')).body
                     if isinstance(node, ast.FunctionDef) and node.name in functions}
            self.assertEqual(found, functions)


if __name__ == '__main__':
    unittest.main()
