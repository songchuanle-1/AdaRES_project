"""Use recorded CPU paths; reject altered self/language or original norm/FFN evidence."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick

CORE = ast.parse((ROOT / 'scripts/y87_observer_core.py').read_text(encoding='utf-8'))
N = {}
CHECK = next(node for node in CORE.body if isinstance(node, ast.FunctionDef) and node.name == 'check_response_rows')
exec(compile(ast.Module(body=[CHECK], type_ignores=[]), '<Y87 recorded-path guard>', 'exec'), N)


class NativePlan(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = next(candidate for candidate in work['candidates'] if candidate['id'] == 'AdaRES-Y87')
        version = tick.load(ROOT / self.c['source_manifest']['path'])
        self.cpu = tick.load(ROOT / version['CPU_result']['path'])
        self.rows = copy.deepcopy(self.cpu['CPU_fusion_paths'])
        self.cpu_program = ROOT / version['CPU_program']['path']

    def check(self, rows=None):
        N['check_response_rows'](self.rows if rows is None else rows)

    def test_actual_CPU_paths_do_not_confer_GPU_PASS(self):
        self.check()
        self.assertTrue(self.cpu['implementation_test_passed'])
        for key in ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward', 'native_sparse_forward', 'admission'):
            self.assertIs(self.cpu[key], False)

    def test_fourteen_callbacks_identical_in_executed_CPU_and_planned_native(self):
        expected = {node.name: ast.dump(node, include_attributes=False)
                    for node in CORE.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(len(expected), 14)
        for path in (self.cpu_program, ROOT / self.c['prepared_native_body']['path']):
            found = {node.name: ast.dump(node, include_attributes=False)
                     for node in ast.parse(path.read_text(encoding='utf-8')).body
                     if isinstance(node, ast.FunctionDef) and node.name in expected}
            self.assertEqual(found, expected)

    def test_sum_not_average_and_original_norm_location_preserved(self):
        source = (ROOT / 'AdaRES/ipdn/model_paper5/adares_y87_self_language_product/dec.py').read_text(encoding='utf-8')
        component = next(node for node in ast.parse(source).body
                         if isinstance(node, ast.ClassDef) and node.name == 'SelfLanguageProductFusion')
        self.assertFalse(any(isinstance(node, ast.Div) for node in ast.walk(component)))
        self.assertIn('self.norm_qqa(self.self_language_product(query_self, query_qla))', source)
        self.assertIn('query_qia = (coarse_query + fine_query)/2', source)
        self.assertEqual((self.cpu['parameters'], self.cpu['additional_parameters']), (164695173, 65536))

    def test_missing_extra_or_reordered_paths_refused(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1], list(reversed(self.rows))):
            with self.assertRaises(AssertionError):
                self.check(rows)

    def test_original_self_language_identity_and_formula_required(self):
        for key in ('original_self_identity', 'original_language_identity', 'both_inputs_undetached',
                    'float64_sum_product_projection_reference'):
            rows = copy.deepcopy(self.rows)
            rows[8][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(rows)

    def test_zero_then_learned_response_required(self):
        for index, key, value in [(0, 'weight_nonzero', True), (0, 'fusion_delta_max', .1),
                                 (8, 'weight_nonzero', False), (8, 'fusion_delta_max', 0.)]:
            rows = copy.deepcopy(self.rows)
            rows[index][key] = value
            with self.assertRaises(AssertionError):
                self.check(rows)

    def test_nonfinite_and_negative_response_refused(self):
        for value in (float('nan'), float('inf'), -1.):
            rows = copy.deepcopy(self.rows)
            rows[8]['fusion_delta_max'] = value
            with self.assertRaises(AssertionError):
                self.check(rows)

    def test_exact_stage_step_and_flag_types_required(self):
        for key, value in [('step', True), ('stage', 2.), ('stage', 5), ('weight_nonzero', 1)]:
            rows = copy.deepcopy(self.rows)
            rows[8][key] = value
            with self.assertRaises(AssertionError):
                self.check(rows)

    def test_parent_anchor_and_actual_original_norm_ffn_consumers(self):
        for key in ('decoder_initial_RNG_and_parent_weights_exact', 'decoder_all_stages_anchor_exact',
                    'decoder_input_gradients_anchor_close', 'original_initial_prediction_unchanged',
                    'original_self_language_and_norm_ffn_consumption_verified', 'bypassed_fusion_negative_control_exact'):
            self.assertIs(self.cpu[key], True)
        for row in self.cpu['CPU_decoder_steps']:
            self.assertEqual((row['fusion_calls'], row['original_norm_consumption_calls'],
                              row['original_ffn_consumption_calls']), (6, 6, 6))


if __name__ == '__main__':
    unittest.main()
