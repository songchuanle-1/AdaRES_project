"""Admission tests require the actual Y89 native result; no fixture can replace it."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
import advance_y89 as advance
import test_y89_execution_guards as execution_guards


class Admission(execution_guards.ExecutionGuards):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = copy.deepcopy(next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y89'))
        plan = tick.load(ROOT / self.c['native_smoke_plan']['path'])
        self.f = ast.literal_eval(ast.parse(
            (ROOT / plan['program']['path']).read_text(encoding='utf-8')).body[0].value)
        result = self.c.get('native_smoke_result')
        assert result, 'Actual native Y89 evidence is required; CPU preparation is not admission'
        self.raw = tick.load(ROOT / result['path'])

    def check(self, value=None):
        advance.validate_result(self.raw if value is None else value, self.c, self.f)

    def test_actual_native_result(self):
        self.check()

    def test_identity_version_yaml_goal(self):
        for key, value in [('id', 'AdaRES-Y88'), ('code_commit', '0' * 40),
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
                'original_hard_gate_algorithm_preserved', 'original_competition_evidence_preserved'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: False}))
        for key in ('validation_used', 'model_weights_saved'):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: True}))

    def test_all_fourteen_original_consumers(self):
        for key in ('private_evidence_norm_calls', 'shared_MLP_calls', 'shared_query_norm_calls',
                    'shared_indicator_calls', 'coarse_gate_consumers'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: 0}))

    def test_per_step_original_gradients_and_updates(self):
        for key, value in [('all_finite', False), ('input_gradients_finite', False),
                ('shared_MLP_parameter_tensors', 0), ('shared_query_norm_parameter_tensors', 0),
                ('evidence_norm_gradient_tensors', 0), ('private_evidence_norm_calls', 0),
                ('coarse_gate_consumers', 0)]:
            changed = copy.deepcopy(self.raw)
            changed['gradient_checks'][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(changed)
        for key in ('norm_gradient_maxima', 'norm_parameter_updates'):
            for value in (0., float('nan'), float('inf')):
                changed = copy.deepcopy(self.raw)
                changed['gradient_checks'][1][key][8] = value
                with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                    self.check(changed)
        changed = copy.deepcopy(self.raw)
        changed['gradient_checks'].reverse()
        with self.assertRaises(AssertionError):
            self.check(changed)

    def test_missing_private_norm_response_rows(self):
        with self.assertRaises(AssertionError):
            self.check(dict(self.raw, norm_paths=self.raw['norm_paths'][:-1]))

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
