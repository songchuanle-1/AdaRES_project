"""Reject missing or altered native Y79 evidence before source selection/admission."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
import advance_y79_server1 as advance
import y79_deployment as deploy


class Admission(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = copy.deepcopy(next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y79'))
        plan = tick.load(ROOT / self.c['native_smoke_plan_server1']['path'])
        self.f = ast.literal_eval(ast.parse((ROOT / plan['program']['path']).read_text(encoding='utf-8')).body[0].value)
        self.raw = tick.load(ROOT / self.c['native_smoke_result']['path'])

    def check(self, raw=None):
        advance.validate_result(self.raw if raw is None else raw, self.c, self.f)

    def test_actual_recorded_native_result(self):
        self.check()

    def test_wrong_identity_version_config_goal_or_yaml(self):
        for key, value in [('id', 'AdaRES-Y77'), ('code_commit', '0'*40), ('config', 'other.yaml'), ('goal_sha256', '0'*64), ('native_YAML_sha256', '0'*64)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_missing_or_nonfinite_losses(self):
        for value in ([float('nan'), 1.], [1., float('inf')], [1.]):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, losses=value))

    def test_synthetic_validation_or_missing_native_execution(self):
        for key, value in [('split', 'val'), ('native_data_loaded', False), ('backward_finite', False), ('intended_path_observed', False), ('native_modes_verified', False), ('validation_used', True), ('model_weights_saved', True)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_all_native_steps_and_consumers_required(self):
        for key in ('steps', 'optimizer_steps', 'shared_head_calls', 'query_norm_consumption_calls', 'feedback_calls', 'qla_consumption_calls', 'next_lla_consumption_calls'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: 0}))

    def test_original_hard_gate_initial_prediction_and_static_mask_required(self):
        for key in ('static_mask_features_preserved', 'original_initial_prediction_preserved'):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: False}))

    def test_exact_previous_mask_identity_undetached_and_formula_required(self):
        for key in ('current_query_identity', 'original_language_ffn_identity', 'original_language_padding_identity', 'both_inputs_undetached', 'padded_output_identity', 'static_mask_features_preserved', 'float64_attention_reference_checked'):
            value = copy.deepcopy(self.raw)
            value['feedback_paths'][7][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(value)

    def test_zero_then_nonzero_response_required(self):
        for index, key, value in [(0, 'weight_nonzero', True), (0, 'feedback_delta_max', .01), (7, 'weight_nonzero', False), (7, 'feedback_delta_max', 0.)]:
            raw = copy.deepcopy(self.raw)
            raw['feedback_paths'][index][key] = value
            with self.assertRaises(AssertionError):
                self.check(raw)

    def test_previous_gradient_and_new_weight_update_required(self):
        for key, value in [('both_input_gradients_finite', False), ('feedback_weight_gradient_max', 0.), ('parameter_update_max', 0.)]:
            raw = copy.deepcopy(self.raw)
            raw['gradient_checks'][1][key] = value
            with self.assertRaises(AssertionError):
                self.check(raw)

    def test_per_step_original_consumers_required(self):
        for key in ('indicator_tensors', 'indicator_calls', 'feedback_calls', 'qla_consumption_calls', 'next_lla_consumption_calls', 'query_norm_consumption_calls'):
            raw = copy.deepcopy(self.raw)
            raw['gradient_checks'][1][key] = 0
            with self.assertRaises(AssertionError):
                self.check(raw)

    def test_original_dataset_and_four_scenes_required(self):
        raw = copy.deepcopy(self.raw)
        raw['data_source']['config_data_root'] = '/wrong/data'
        with self.assertRaises(AssertionError):
            self.check(raw)
        raw = copy.deepcopy(self.raw)
        raw['batches'][1] = copy.deepcopy(raw['batches'][0])
        with self.assertRaises(AssertionError):
            self.check(raw)

    def test_unknown_foreign_or_changed_power_refused(self):
        for gpu, compute in [('nan,0,350', ''), ('0,0,280', ''), ('0,0,390', ''), ('0,2000,350', ''), ('0,0,350', '99')]:
            with self.assertRaises(AssertionError):
                advance.idle(dict(gpu=gpu, compute=compute), 350.)

    def test_slot_ownership_and_queued_work_refused(self):
        pause = dict(id='AdaRES-Y57', status='paused', gpu=1, dispatch_hold=True, resume_requires_user_authorization=True)
        queue = dict(gpu_queues={'server-2': [pause], 'server-1': []})
        advance.unreserved(queue, {}, 'server-1', 0)
        for status in ('waiting', 'claiming', 'running'):
            value = copy.deepcopy(queue)
            value['gpu_queues']['server-1'].append(dict(id='foreign', status=status, gpu=0))
            with self.assertRaises(AssertionError):
                advance.unreserved(value, {}, 'server-1', 0)
        with self.assertRaises(AssertionError):
            advance.unreserved(queue, {'server-1': {'0': 'foreign'}}, 'server-1', 0)

    def test_y57_pause_required(self):
        pause = dict(id='AdaRES-Y57', status='running', dispatch_hold=True, resume_requires_user_authorization=True)
        with self.assertRaises(AssertionError):
            advance.unreserved(dict(gpu_queues={'server-2': [pause]}), {}, 'server-1', 0)

    def test_exact_chain_preserves_both_original_running_models(self):
        for cid, commit in [('AdaRES-Y77', deploy.Y77), ('AdaRES-Y78', deploy.Y78)]:
            self.assertIn(deploy.Y79, deploy.contract(commit, deploy.Y79, cid))

    def test_unknown_or_reversed_chain_refused(self):
        for commit, head in [(deploy.Y77, '0'*40), ('0'*40, deploy.Y79), (deploy.Y79, deploy.Y77)]:
            with self.assertRaises(AssertionError):
                deploy.contract(commit, head, 'AdaRES-Y77')

    def test_only_seven_exact_additions_allowed(self):
        rows = [['A', path] for path in deploy.paths('AdaRES-Y79')]
        deploy.validate_delta(rows, 'AdaRES-Y79')
        for bad in (rows + [['A', 'other.py']], [['M', rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, 'AdaRES-Y79')


if __name__ == '__main__':
    unittest.main()
