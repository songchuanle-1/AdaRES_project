"""Validate the actual original Y88 native evidence, including negative controls."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
import advance_y88 as advance
import y88_deployment as deploy


class Admission(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = copy.deepcopy(next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y88'))
        plan = tick.load(ROOT / self.c['native_smoke_plan']['path'])
        self.f = ast.literal_eval(ast.parse(
            (ROOT / plan['program']['path']).read_text(encoding='utf-8')).body[0].value)
        result = self.c.get('native_smoke_result')
        assert result, 'Actual native evidence is required for this admission suite'
        self.raw = tick.load(ROOT / result['path'])

    def check(self, raw=None):
        advance.validate_result(self.raw if raw is None else raw, self.c, self.f)

    def test_actual_native_record_required(self):
        self.check()

    def test_identity_version_yaml_and_goal_required(self):
        for key, value in [('id', 'AdaRES-Y86'), ('code_commit', '0' * 40),
                           ('config', 'other.yaml'), ('goal_sha256', '0' * 64),
                           ('native_YAML_sha256', '0' * 64)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_native_train_and_finite_losses_required(self):
        for key, value in [('split', 'val'), ('native_entry', 'synthetic'),
                ('steps', 1), ('optimizer_steps', 1), ('losses', [1.]),
                ('losses', [1., float('nan')]), ('losses', [1., float('inf')])]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_actual_execution_and_original_paths_required(self):
        for key in ('native_data_loaded', 'backward_finite', 'intended_path_observed',
                    'native_modes_verified', 'static_mask_features_preserved',
                    'original_hard_gate_algorithm_preserved', 'static_mask_features_preserved'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: False}))
        for key in ('validation_used', 'model_weights_saved'):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: True}))

    def test_shared_head_and_original_norm_ffn_counts_required(self):
        for key in ('shared_head_calls', 'query_norm_consumption_calls', 'projection_calls',
                    'original_mask_consumers', 'original_score_consumers'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: 0}))

    def test_original_self_language_and_undetached_formula_required(self):
        for key in ('original_normalized_query_identity', 'original_mask_consumer',
                    'input_undetached', 'float64_quadratic_projection_reference'):
            value = copy.deepcopy(self.raw)
            value['projection_paths'][8][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(value)

    def test_initial_zero_and_learned_response_required(self):
        for index, key, entry in [(0, 'weight_nonzero', True), (0, 'delta_max', .1),
                                 (8, 'weight_nonzero', False), (8, 'delta_max', 0.)]:
            value = copy.deepcopy(self.raw)
            value['projection_paths'][index][key] = entry
            with self.assertRaises(AssertionError):
                self.check(value)

    def test_nonfinite_or_missing_fusion_rows_refused(self):
        for entry in (float('nan'), float('inf'), -1.):
            value = copy.deepcopy(self.raw)
            value['projection_paths'][8]['delta_max'] = entry
            with self.assertRaises(AssertionError):
                self.check(value)
        for rows in (self.raw['projection_paths'][:-1], list(reversed(self.raw['projection_paths']))):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, projection_paths=rows))

    def test_original_and_new_gradients_updates_required(self):
        for key, entry in [('input_gradients_finite', False), ('all_finite', False),
                           ('new_gradient_max', 0.), ('new_gradient_max', float('nan')),
                           ('parameter_update_max', 0.), ('parameter_update_max', float('inf'))]:
            value = copy.deepcopy(self.raw)
            value['gradient_checks'][1][key] = entry
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(value)

    def test_per_step_counts_and_order_required(self):
        for key in ('indicator_tensors', 'indicator_calls', 'query_norm_consumption_calls',
                    'projection_calls', 'original_mask_consumers', 'original_score_consumers', 'tensors'):
            value = copy.deepcopy(self.raw)
            value['gradient_checks'][1][key] = 0
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(value)
        value = copy.deepcopy(self.raw)
        value['gradient_checks'].reverse()
        with self.assertRaises(AssertionError):
            self.check(value)

    def test_exact_TRAIN_dataset_and_four_scenes_required(self):
        for key, entry in [('config_data_root', '/wrong/data'), ('dataset_class', 'synthetic')]:
            value = copy.deepcopy(self.raw)
            value['data_source'][key] = entry
            with self.assertRaises(AssertionError):
                self.check(value)
        value = copy.deepcopy(self.raw)
        value['batches'][1] = copy.deepcopy(value['batches'][0])
        with self.assertRaises(AssertionError):
            self.check(value)

    def test_memory_and_original_power_required(self):
        for peak in (0, 24 * 2**30, float('nan')):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, peak_reserved_bytes=peak))
        advance.idle(dict(gpu='0,1,350', compute=''), 350.)
        for gpu, compute in [('nan,0,350', ''), ('0,0,390', ''), ('0,2000,350', ''), ('0,0,350', '99')]:
            with self.assertRaises(AssertionError):
                advance.idle(dict(gpu=gpu, compute=compute), 350.)

    def test_reservations_and_pause_preserved(self):
        paused = dict(id='AdaRES-Y57', status='paused', gpu=1,
                      dispatch_hold=True, resume_requires_user_authorization=True)
        queue = dict(gpu_queues={'server-2': [paused]})
        advance.unreserved(queue, {}, 'server-2', 0)
        for status in ('waiting', 'claiming', 'running'):
            changed = copy.deepcopy(queue)
            changed['gpu_queues']['server-2'].append(dict(id='foreign', status=status, gpu=0))
            with self.assertRaises(AssertionError):
                advance.unreserved(changed, {}, 'server-2', 0)
        with self.assertRaises(AssertionError):
            advance.unreserved(queue, {'server-2': {'0': 'foreign'}}, 'server-2', 0)
        queue['gpu_queues']['server-2'][0]['status'] = 'running'
        with self.assertRaises(AssertionError):
            advance.unreserved(queue, {}, 'server-2', 0)

    def test_seven_independent_additions_only(self):
        rows = [['A', path] for path in deploy.paths('AdaRES-Y88')]
        deploy.validate_delta(rows, 'AdaRES-Y88')
        self.assertNotIn('AdaRES-Y86', deploy.ADDITIONS.values())
        for bad in (rows + [['A', 'other.py']], [['M', rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, 'AdaRES-Y88')

    def test_source_chain_preserves_Y84_Y85_and_refuses_old_Y88(self):
        for cid, commit in [('AdaRES-Y84', deploy.Y84), ('AdaRES-Y85', deploy.Y85)]:
            self.assertIn(deploy.Y88, deploy.contract(commit, deploy.Y88, cid))
        for commit, head in [('0' * 40, deploy.Y88), (deploy.Y88, deploy.Y85),
                             ('053aea85a03be4244879c90362455c45327a7d64', deploy.Y88)]:
            with self.assertRaises(AssertionError):
                deploy.contract(commit, head, 'AdaRES-Y84')


if __name__ == '__main__':
    unittest.main()
