"""Require actual Y93 GPU-native hidden-gate output before any admission."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
import advance_y93 as advance
import test_y93_execution_guards as execution_guards

class Admission(execution_guards.ExecutionGuards):

    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = copy.deepcopy(next((row for row in work['candidates'] if row['id'] == 'AdaRES-Y93')))
        plan = tick.load(ROOT / self.c['native_smoke_plan']['path'])
        self.f = ast.literal_eval(ast.parse((ROOT / plan['program']['path']).read_text(encoding='utf-8')).body[0].value)
        result = self.c.get('native_smoke_result')
        assert result, 'Actual Y93 native evidence is mandatory; CPU preparation cannot admit'
        self.raw = tick.load(ROOT / result['path'])

    def check(self, value=None):
        advance.validate_result(self.raw if value is None else value, self.c, self.f)

    def test_actual_native_result_and_chosen_runtime(self):
        self.check()
        candidate, _, _, frozen = advance.prepared(tick.load(ROOT / 'pipeline/research_work.json'))
        self.assertEqual(candidate['required_server'], 'server-2')
        self.assertEqual(frozen['runtime_power_limit_watts'], 280.0)

    def test_identity_version_yaml_goal(self):
        for key, value in (('id', 'AdaRES-Y92'), ('code_commit', '0' * 40), ('config', 'other.yaml'), ('goal_sha256', '0' * 64), ('native_YAML_sha256', '0' * 64)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_native_train_and_finite_two_losses(self):
        for key, value in (('split', 'val'), ('native_entry', 'synthetic'), ('steps', 1), ('optimizer_steps', 1), ('losses', [1.0]), ('losses', [1.0, float('nan')]), ('losses', [1.0, float('inf')])):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_native_modes_original_competition_and_no_validation(self):
        for key in ('native_data_loaded', 'backward_finite', 'intended_path_observed', 'native_modes_verified', 'static_mask_features_preserved', 'original_hard_gate_algorithm_preserved', 'original_competition_and_evidence_preserved'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: False}))
        for key in ('validation_used', 'model_weights_saved', 'admission'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: True}))

    def test_all_original_hidden_and_linear_consumers(self):
        for key in ('evidence_skip_calls', 'first_hidden_consumers', 'second_hidden_consumers', 'final_linear_consumers', 'shared_query_norm_calls', 'original_evidence_norm_calls', 'original_overlap_consumers', 'original_MLP_layer_calls', 'coarse_gate_consumers'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: 0}))

    def test_per_step_original_gradients_and_updates(self):
        for key, value in (('all_finite', False), ('input_gradients_finite', False), ('original_head_parameter_tensors', 0), ('evidence_skip_calls', 0), ('original_MLP_layer_calls', 0), ('indicator_calls', 0), ('new_gradient_max', 0.0), ('parameter_update_max', float('nan'))):
            changed = copy.deepcopy(self.raw)
            changed['gradient_checks'][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(changed)
        changed = copy.deepcopy(self.raw)
        changed['gradient_checks'].reverse()
        with self.assertRaises(AssertionError):
            self.check(changed)

    def test_hidden_paths_and_exact_initial_anchor_required(self):
        with self.assertRaises(AssertionError):
            self.check(dict(self.raw, hidden_paths=self.raw['hidden_paths'][:-1]))
        for key, value in (('original_query_identity', False), ('original_mass_identity', False), ('original_evidence_exact', False), ('evidence_and_second_hidden_identity', False), ('both_inputs_undetached', False), ('float64_residual_reference', False), ('final_linear_consumes_residual_hidden', False), ('projection_delta_max', 0.0), ('hidden_delta_max', 0.0)):
            changed = copy.deepcopy(self.raw)
            changed['hidden_paths'][8][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(changed)
        changed = copy.deepcopy(self.raw)
        changed['hidden_paths'][0]['hidden_delta_max'] = 1.0
        with self.assertRaises(AssertionError):
            self.check(changed)

    def test_exact_native_TRAIN_data_and_four_distinct_scenes(self):
        for key, value in (('config_data_root', '/wrong/data'), ('dataset_class', 'synthetic')):
            changed = copy.deepcopy(self.raw)
            changed['data_source'][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(changed)
        changed = copy.deepcopy(self.raw)
        changed['batches'][1] = copy.deepcopy(changed['batches'][0])
        with self.assertRaises(AssertionError):
            self.check(changed)

    def test_native_memory_bound(self):
        for value in (0, 24 * 2 ** 30, float('nan')):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, peak_reserved_bytes=value))
if __name__ == '__main__':
    unittest.main()
