"""Y85 plan guards use actual CPU observations without claiming GPU execution."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick

BODY = (ROOT / 'scripts/probe_y85_native_smoke.py').read_text(encoding='utf-8')
TREE = ast.parse(BODY)
CHECK = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'check_mask_norm_rows')
N = {}
exec(compile(ast.Module(body=[CHECK], type_ignores=[]), '<Y85 observed-path guard>', 'exec'), N)


class NativePlan(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y85')
        version = tick.load(ROOT / candidate['source_manifest']['path'])
        self.cpu = tick.load(ROOT / version['CPU_result']['path'])
        self.rows = copy.deepcopy(self.cpu['CPU_norm_paths'])
        snapshot = tick.load(ROOT / version['source_snapshot']['path'])
        self.cpu_tree = ast.parse((ROOT / snapshot['CPU_program']['path']).read_text(encoding='utf-8'))

    def check(self, rows=None):
        N['check_mask_norm_rows'](self.rows if rows is None else rows)

    def test_actual_CPU_rows_are_not_native_GPU_PASS(self):
        self.check()
        self.assertTrue(self.cpu['implementation_test_passed'])
        self.assertFalse(self.cpu['native_task_backward'])
        self.assertFalse(self.cpu['native_smoke_passed'])

    def test_seven_callbacks_match_actually_executed_CPU_source(self):
        core = ast.parse((ROOT / 'scripts/y85_observer_core.py').read_text(encoding='utf-8'))
        expected = {n.name: ast.dump(n, include_attributes=False) for n in core.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(len(expected), 7)
        for tree in (TREE, self.cpu_tree):
            self.assertEqual({n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in expected}, expected)

    def test_missing_extra_and_reversed_paths_refused(self):
        for rows in (self.rows[:1], self.rows + self.rows[:1], list(reversed(self.rows))):
            with self.assertRaises(AssertionError): self.check(rows)

    def test_original_raw_identity_undetached_and_module_required(self):
        for key in ('original_x_mask_output_identity', 'input_undetached', 'shared_module_identity'):
            rows = copy.deepcopy(self.rows); rows[1][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError): self.check(rows)

    def test_all_channels_original_epsilon_and_reference_required(self):
        for key in ('all_channels_and_original_epsilon', 'float64_LayerNorm_reference'):
            rows = copy.deepcopy(self.rows); rows[1][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError): self.check(rows)

    def test_exact_step_type_and_boolean_flags_required(self):
        for key, value in [('step', True), ('step', 1.), ('step', 0), ('input_undetached', 1)]:
            rows = copy.deepcopy(self.rows); rows[1][key] = value
            with self.assertRaises(AssertionError): self.check(rows)

    def test_CPU_bypass_anchor_does_not_claim_active_W2_equivalence(self):
        for key in ('norm_bypass_all_stages_W2_anchor_exact', 'norm_bypass_input_gradients_W2_anchor_close', 'active_transform_changes_intended_initial_masks', 'bypass_restoration_exact', 'padded_batched_features_zero'):
            self.assertIs(self.cpu[key], True)
        self.assertEqual(self.cpu['additional_parameters'], 512)
        self.assertEqual({row['dtype'] for row in self.cpu['component_controls']}, {'torch.float32', 'torch.float64'})

    def test_actual_native_train_entry_is_planned_once(self):
        calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'native' and n.func.attr == 'train']
        self.assertEqual(len(calls), 1)
        self.assertEqual([ast.unparse(a) for a in calls[0].args], ['1', 'model', 'FirstTwoBatches()', 'optimizer', 'scheduler', 'cfg', 'logger', 'Writer()'])


if __name__ == '__main__':
    unittest.main()
