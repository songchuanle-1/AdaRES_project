"""Actual Y90 GPU-native output is mandatory; CPU fixtures cannot admit it."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
import advance_y90 as advance
import test_y90_execution_guards as execution_guards


class Admission(execution_guards.ExecutionGuards):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = copy.deepcopy(next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y90'))
        plan = tick.load(ROOT / self.c['native_smoke_plan']['path'])
        self.f = ast.literal_eval(ast.parse(
            (ROOT / plan['program']['path']).read_text(encoding='utf-8')).body[0].value)
        result = self.c.get('native_smoke_result')
        assert result, 'Actual native Y90 evidence is required; CPU preparation is not admission'
        self.raw = tick.load(ROOT / result['path'])

    def check(self, value=None):
        advance.validate_result(self.raw if value is None else value, self.c, self.f)

    def test_actual_native_result_and_chosen_runtime(self):
        self.check()
        candidate, _, _, frozen = advance.prepared(
            tick.load(ROOT / 'pipeline/research_work.json'))
        self.assertEqual(candidate['required_server'], 'server-1')
        self.assertEqual(frozen['runtime_power_limit_watts'], 350.)

    def test_identity_version_yaml_goal(self):
        for key, value in [('id', 'AdaRES-Y89'), ('code_commit', '0' * 40),
                ('config', 'other.yaml'), ('goal_sha256', '0' * 64), ('native_YAML_sha256', '0' * 64)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_native_train_and_finite_two_losses(self):
        for key, value in [('split', 'val'), ('native_entry', 'synthetic'), ('steps', 1),
                ('optimizer_steps', 1), ('losses', [1.]), ('losses', [1., float('nan')]),
                ('losses', [1., float('inf')])]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_native_modes_original_competition_and_no_validation(self):
        for key in ('native_data_loaded', 'backward_finite', 'intended_path_observed',
                'native_modes_verified', 'static_mask_features_preserved',
                'original_hard_gate_algorithm_preserved', 'original_competition_overlap_and_max_preserved'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: False}))
        for key in ('validation_used', 'model_weights_saved'):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: True}))

    def test_all_fourteen_original_consumers(self):
        for key in ('peer_gate_calls', 'original_overlap_consumers', 'weighted_context_consumers',
                    'shared_MLP_calls', 'shared_query_norm_calls', 'original_evidence_norm_calls',
                    'coarse_gate_consumers'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: 0}))

    def test_per_step_original_gradients_and_updates(self):
        for key, value in [('all_finite', False), ('input_gradients_finite', False),
                ('original_head_parameter_tensors', 0), ('peer_gate_calls', 0),
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

    def test_original_affinity_consumers_and_initial_anchor_required(self):
        with self.assertRaises(AssertionError):
            self.check(dict(self.raw, affinity_paths=self.raw['affinity_paths'][:-1]))
        for key, value in [('original_query_identity', False), ('original_mass_identity', False),
                ('normalized_query_input_undetached', False),
                ('float64_bilinear_gate_reference', False), ('weighted_peer_context_consumed', False),
                ('original_overlap_and_max_preserved', False), ('gate_delta_max', 0.)]:
            changed = copy.deepcopy(self.raw)
            changed['affinity_paths'][8][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(changed)
        changed = copy.deepcopy(self.raw)
        changed['affinity_paths'][0]['context_delta_max'] = 1.
        with self.assertRaises(AssertionError):
            self.check(changed)

    def test_learned_context_requires_aggregate_positive_response(self):
        changed = copy.deepcopy(self.raw)
        for row in changed['affinity_paths'][7:]:
            row['context_delta_max'] = 0.
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
