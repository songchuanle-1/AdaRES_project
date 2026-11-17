"""Guard fixtures use actual CPU observations and never claim native GPU execution."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick

TREE = ast.parse((ROOT / 'scripts/y86_observer_core.py').read_text(encoding='utf-8'))
CHECK = next(node for node in TREE.body if isinstance(node, ast.FunctionDef)
             and node.name == 'check_shortcut_rows')
N = {}
exec(compile(ast.Module(body=[CHECK], type_ignores=[]), '<Y86 recorded-path guard>', 'exec'), N)


class NativePlan(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == 'AdaRES-Y86')
        version = tick.load(ROOT / candidate['source_manifest']['path'])
        self.cpu = tick.load(ROOT / version['CPU_result']['path'])
        self.rows = copy.deepcopy(self.cpu['CPU_shortcut_paths'])
        self.program = ROOT / version['CPU_program']['path']

    def check(self, rows=None):
        N['check_shortcut_rows'](self.rows if rows is None else rows)

    def test_recorded_CPU_controls_are_not_native_GPU_PASS(self):
        self.check()
        self.assertTrue(self.cpu['implementation_test_passed'])
        for key in ('native_task_backward', 'native_sparse_forward', 'native_smoke_passed', 'CUDA_initialized'):
            self.assertIs(self.cpu[key], False)

    def test_exact_seven_callbacks_actually_used_by_CPU(self):
        names = {node.name: ast.dump(node, include_attributes=False) for node in TREE.body
                 if isinstance(node, ast.FunctionDef)}
        found = {node.name: ast.dump(node, include_attributes=False)
                 for node in ast.parse(self.program.read_text(encoding='utf-8')).body
                 if isinstance(node, ast.FunctionDef) and node.name in names}
        self.assertEqual(len(names), 7)
        self.assertEqual(found, names)

    def test_two_new_parameters_and_full_model_required(self):
        self.assertEqual((self.cpu['additional_parameters'], self.cpu['parameters']), (2, 164629639))
        self.assertEqual(self.cpu['synthetic_CPU_decoder_optimizer_steps'], 2)

    def test_missing_extra_or_reordered_paths_refused(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1], list(reversed(self.rows))):
            with self.assertRaises(AssertionError):
                self.check(rows)

    def test_original_readout_overlap_and_input_identity_required(self):
        for key in ('original_logits_identity', 'original_all_peer_max_overlap_reference',
                    'no_new_overlap_or_detach', 'both_inputs_undetached', 'float64_scalar_projection_reference'):
            rows = copy.deepcopy(self.rows)
            rows[8][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(rows)

    def test_initial_zero_and_next_step_response_required(self):
        for index, key, value in [(0, 'weight_nonzero', True), (3, 'confidence_delta_max', .1),
                                 (8, 'weight_nonzero', False), (10, 'confidence_delta_max', 0.)]:
            rows = copy.deepcopy(self.rows)
            rows[index][key] = value
            with self.assertRaises(AssertionError):
                self.check(rows)

    def test_nonfinite_or_negative_response_refused(self):
        for value in (float('nan'), float('inf'), -1.):
            rows = copy.deepcopy(self.rows)
            rows[8]['confidence_delta_max'] = value
            with self.assertRaises(AssertionError):
                self.check(rows)

    def test_exact_stage_step_and_flag_types_required(self):
        for key, value in [('step', True), ('stage', 1.), ('stage', 6), ('weight_nonzero', 1)]:
            rows = copy.deepcopy(self.rows)
            rows[8][key] = value
            with self.assertRaises(AssertionError):
                self.check(rows)

    def test_parent_anchors_and_confidence_only_intervention_required(self):
        for key in ('decoder_initial_RNG_and_parent_weights_exact', 'decoder_all_stages_anchor_exact',
                    'decoder_input_gradients_anchor_close', 'nonzero_confidence_only_intervention_restored',
                    'original_static_masks_and_hard_gates_verified'):
            self.assertIs(self.cpu[key], True)


if __name__ == '__main__':
    unittest.main()
