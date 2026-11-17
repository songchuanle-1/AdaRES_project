"""Require actual Y92 visual-side native evidence; never admit CPU fixtures."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
import advance_y92 as advance
import test_y92_execution_guards as execution_guards


class Admission(execution_guards.ExecutionGuards):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = copy.deepcopy(next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y92'))
        plan = tick.load(ROOT / self.c['native_smoke_plan']['path'])
        self.f = ast.literal_eval(ast.parse(
            (ROOT / plan['program']['path']).read_text(encoding='utf-8')).body[0].value)
        result = self.c.get('native_smoke_result')
        assert result, 'Actual Y92 native evidence is required; CPU preparation is not admission'
        self.raw = tick.load(ROOT / result['path'])

    def check(self, value=None):
        advance.validate_result(self.raw if value is None else value, self.c, self.f)

    def test_actual_native_result_and_chosen_runtime(self):
        self.check()
        candidate, _, _, frozen = advance.prepared(tick.load(ROOT / 'pipeline/research_work.json'))
        self.assertEqual(candidate['required_server'], 'server-1')
        self.assertEqual(frozen['runtime_power_limit_watts'], 350.)

    def test_identity_version_yaml_goal(self):
        for key, value in [('id', 'AdaRES-Y91'), ('code_commit', '0' * 40),
                          ('config', 'other.yaml'), ('goal_sha256', '0' * 64),
                          ('native_YAML_sha256', '0' * 64)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_native_train_and_finite_two_losses(self):
        for key, value in [('split', 'val'), ('native_entry', 'synthetic'), ('steps', 1),
                          ('optimizer_steps', 1), ('losses', [1.]),
                          ('losses', [1., float('nan')]), ('losses', [1., float('inf')])]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_native_modes_consumers_and_no_validation(self):
        for key in ('native_data_loaded', 'backward_finite', 'intended_path_observed',
                    'native_modes_verified', 'shared_transformed_visual_mask_features_consumed',
                    'original_hard_gate_algorithm_preserved'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: False}))
        for key in ('validation_used', 'model_weights_saved', 'admission'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: True}))

    def test_projection_and_all_fourteen_original_consumers(self):
        for key in ('projection_calls', 'original_mask_consumers', 'original_score_consumers',
                    'shared_head_calls', 'query_norm_consumption_calls', 'coarse_gate_consumers'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: 0}))

    def test_per_step_original_gradients_and_updates(self):
        for key, value in [('all_finite', False), ('input_gradients_finite', False),
                          ('indicator_tensors', 0), ('projection_calls', 0),
                          ('coarse_gate_consumers', 0), ('indicator_calls', 0),
                          ('new_gradient_max', 0.), ('parameter_update_max', float('nan'))]:
            changed = copy.deepcopy(self.raw)
            changed['gradient_checks'][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(changed)
        changed = copy.deepcopy(self.raw)
        changed['gradient_checks'].reverse()
        with self.assertRaises(AssertionError):
            self.check(changed)

    def test_original_projection_paths_and_initial_anchor_required(self):
        with self.assertRaises(AssertionError):
            self.check(dict(self.raw, projection_paths=self.raw['projection_paths'][:-1]))
        for key, value in [('original_x_mask_output_identity', False), ('input_undetached', False),
                          ('float64_quadratic_projection_reference', False),
                          ('shared_module_identity', False), ('all_seven_original_mask_consumers', False),
                          ('delta_max', 0.)]:
            changed = copy.deepcopy(self.raw)
            changed['projection_paths'][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(changed)
        changed = copy.deepcopy(self.raw)
        changed['projection_paths'][0]['delta_max'] = 1.
        with self.assertRaises(AssertionError):
            self.check(changed)

    def test_exact_native_TRAIN_data_and_four_distinct_scenes(self):
        for key, value in [('config_data_root', '/wrong/data'), ('dataset_class', 'synthetic')]:
            changed = copy.deepcopy(self.raw)
            changed['data_source'][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(changed)
        changed = copy.deepcopy(self.raw)
        changed['batches'][1] = copy.deepcopy(changed['batches'][0])
        with self.assertRaises(AssertionError):
            self.check(changed)

    def test_native_memory_bound(self):
        for value in (0, 24 * 2**30, float('nan')):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, peak_reserved_bytes=value))


if __name__ == '__main__':
    unittest.main()
