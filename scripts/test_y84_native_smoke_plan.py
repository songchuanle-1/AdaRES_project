"""Plan guards against false native claims; fixtures use actual CPU observations."""
import ast
import copy
from pathlib import Path
import unittest
from prepare_y60_stage_heads import ROOT, tick

BODY = (ROOT / 'scripts/probe_y84_native_smoke.py').read_text(encoding='utf-8')
TREE = ast.parse(BODY)
CHECK = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'check_initialization_rows')
N = {}
exec(compile(ast.Module(body=[CHECK], type_ignores=[]), '<Y84 observed-path guard>', 'exec'), N)


class NativePlan(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        c = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y84')
        version = tick.load(ROOT / c['source_manifest']['path'])
        self.cpu = tick.load(ROOT / version['CPU_result']['path'])
        self.rows = copy.deepcopy(self.cpu['CPU_initialization_paths'])
        self.cpu_source = (ROOT / version['source_snapshot']['path'])
        snapshot = tick.load(self.cpu_source)
        self.tree = ast.parse((ROOT / snapshot['CPU_program']['path']).read_text(encoding='utf-8'))

    def check(self, rows=None):
        N['check_initialization_rows'](self.rows if rows is None else rows)

    def test_actual_CPU_rows_only_not_native_PASS(self):
        self.check()
        self.assertTrue(self.cpu['implementation_test_passed'])
        self.assertFalse(self.cpu['native_task_backward'])
        self.assertFalse(self.cpu['native_smoke_passed'])

    def test_six_callbacks_identical_to_actually_executed_CPU_source(self):
        core = ast.parse((ROOT / 'scripts/y84_observer_core.py').read_text(encoding='utf-8'))
        expected = {n.name: ast.dump(n, include_attributes=False) for n in core.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(len(expected), 6)
        for tree in (TREE, self.tree):
            found = {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in expected}
            self.assertEqual(found, expected)

    def test_missing_or_reordered_rows_refused(self):
        for rows in (self.rows[:1], list(reversed(self.rows))):
            with self.assertRaises(AssertionError): self.check(rows)

    def test_original_identities_reference_and_undetached_inputs_required(self):
        for key in ('original_generated_query_identity', 'original_scene_visual_identity', 'original_padding_and_point_mass_identity', 'all_inputs_undetached', 'float64_weighted_context_reference'):
            rows = copy.deepcopy(self.rows); rows[1][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError): self.check(rows)

    def test_initial_zero_anchor_required(self):
        for key, value in [('weight_nonzero', True), ('query_delta_max', .1)]:
            rows = copy.deepcopy(self.rows); rows[0][key] = value
            with self.assertRaises(AssertionError): self.check(rows)

    def test_next_step_learned_response_required(self):
        for key, value in [('weight_nonzero', False), ('query_delta_max', 0.)]:
            rows = copy.deepcopy(self.rows); rows[1][key] = value
            with self.assertRaises(AssertionError): self.check(rows)

    def test_nonfinite_or_negative_response_refused(self):
        for value in (float('nan'), float('inf'), -1.):
            rows = copy.deepcopy(self.rows); rows[1]['query_delta_max'] = value
            with self.assertRaises(AssertionError): self.check(rows)

    def test_stage_and_weight_flags_are_exact_types(self):
        for key, value in [('step', True), ('step', 0.), ('weight_nonzero', 1)]:
            rows = copy.deepcopy(self.rows); rows[1][key] = value
            with self.assertRaises(AssertionError): self.check(rows)


if __name__ == '__main__':
    unittest.main()
